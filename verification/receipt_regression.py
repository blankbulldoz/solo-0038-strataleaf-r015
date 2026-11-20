"""Independent, re-runnable regression for StrataLeaf idempotent request receipts.

Run from the repository root:

    python3 -m verification.receipt_regression

The script uses only a temporary directory, prints per-step progress to stderr
and one JSON summary line to stdout, and exits non-zero on the first violated
expectation.

Why this can decide replay vs conflict independently
----------------------------------------------------
Every verdict is derived in this file from the *raw timeline of submitted
requests* by ``TimelineModel``: the first successful commit binds a request key
to a type-tagged canonical identity (collection, document key, put/delete,
body, expected revision, expires_at, and operation order).  A later call under
that key replays only when the canonical identity is equal; otherwise it must
conflict.  Documents are canonicalized with explicit tags for null vs missing,
bool vs int, and ordered arrays, so ``true`` never equals ``1``, ``[1, 2]``
never equals ``[2, 1]``, and an absent member never equals an explicit null.
Only JSON *object member order* is ignored, which is the sole equivalence the
wire format promises.  Numeric expiry int/float spelling is kept distinct for
identity, matching the serialized request; the stored expiry is compared
numerically because the SQLite column has REAL affinity.

The script never imports or calls ``codec.digest``, ``request_hash``, the
receipt lookup branch, or any product helper to *decide* the expected verdict.
It also does not use "the current body now matches" or "the current revision
looks right" as a substitute for identity.

What the checks rule out
------------------------
* accepting a changed expires_at, expected revision, operation order, body,
  array order, bool/int spelling, or absent-vs-null member as a replay;
* accepting a request for a different document key or collection merely
  because its body matches, or a delete/put swap with the same key;
* using "the current body happens to equal the requested body" as identity;
* replaying by re-executing prerequisites (a stale expected or changed body
  would then fail or rewrite state);
* rewriting current bodies, document versions, expiry facts, full commit rows,
  receipt rows, or the head during a replay or conflict;
* recovering replay evidence from retained history: after compaction removes
  every full commit and superseded version, and after closing/reopening, legal
  old requests still replay and different ones still conflict;
* accepting a different request under the same key by keeping history,
  clearing receipts, or swapping request keys (keys are pinned before/after);
* failed first attempts (version, schema/type, unique, replica-rejected)
  occupying a receipt or leaving partial rows;
* keyless writes being deduplicated, and genuine new requests/rejections no
  longer working.

What it does not cover
----------------------
* SHA-256 collision resistance of the product's request fingerprint (assumed);
* concurrent writers racing on the same request key from multiple threads or
  processes, or a crash mid-transaction;
* backup/restore round trips of receipt rows, replicas applying request keys,
  ``transfer.import_lines`` with a request key, expiry sweeps, and request-key
  syntax/length limits (they share ``WriteBatch`` but are not exercised here);
* physical contents of rows removed below the retention floor, beyond asserting
  that commits/versions were actually removed;
* performance, WAL checkpointing, and SQLite busy-timeout behaviour.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import traceback
from pathlib import Path

from strataleaf import Archives, Conflict, Database, InvalidDocument, SnapshotTooOld
from strataleaf.errors import ReadOnly

COLLECTION = "delivery"


# ---------------------------------------------------------------------------
# Independent request identity model
# ---------------------------------------------------------------------------
# JSON values are compared with explicit type tags so that true and 1, an
# ordered list and a reordered list, and an absent member and an explicit null
# can never collapse into the same identity.  Object members are sorted, which
# is the only equivalence the wire format promises (JSON object member order is
# not semantic).  Lists keep their order.

def canon(value):
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, list):
        return ("list", tuple(canon(item) for item in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((member, canon(item)) for member, item in value.items())))
    raise TypeError("unsupported value in test request: " + repr(value))


def canon_expected(value):
    if value is None:
        return ("no-expectation",)
    if isinstance(value, bool):
        return ("expected-bool", value)
    if isinstance(value, int):
        return ("expected-int", value)
    raise TypeError("invalid expected in test request: " + repr(value))


def canon_expiry(value):
    # SQLite's expiry column has REAL affinity, and any finite numeric
    # timestamp denotes the same instant whether written as 100 or 100.0.
    return ("no-expiry",) if value is None else ("expiry", float(value))


def request_expiry(value):
    # Request identity follows the JSON wire representation the batch records:
    # an integer timestamp and a float timestamp are different request text.
    if value is None:
        return ("no-expiry",)
    if isinstance(value, bool):
        raise TypeError("invalid expires_at in test request: " + repr(value))
    if isinstance(value, int):
        return ("int-expiry", value)
    if isinstance(value, float):
        return ("float-expiry", repr(value))
    raise TypeError("invalid expires_at in test request: " + repr(value))


def op_identity(operation):
    if operation["kind"] == "delete":
        body = ("delete",)
    else:
        body = ("put", canon(operation["document"]))
    return (
        "operation",
        operation["collection"],
        operation["key"],
        body,
        canon_expected(operation["expected"]),
        request_expiry(operation["expires_at"]),
    )


def request_identity(operations):
    # Operation order is part of the request: the same writes in a different
    # order are a different request even though the final bodies would agree.
    return tuple(op_identity(operation) for operation in operations)


def put_op(collection, key, document, expected=None, expires_at=None):
    return {
        "kind": "put",
        "collection": collection,
        "key": key,
        "document": document,
        "expected": expected,
        "expires_at": expires_at,
    }


def delete_op(collection, key, expected=None):
    return {
        "kind": "delete",
        "collection": collection,
        "key": key,
        "document": None,
        "expected": expected,
        "expires_at": None,
    }


def product_operations(operations):
    """The operation shape stored in the full commit payload."""
    return [
        {
            "collection": operation["collection"],
            "key": operation["key"],
            "document": operation["document"],
            "expected": operation["expected"],
            "expires_at": operation["expires_at"],
        }
        for operation in operations
    ]


class TimelineModel:
    """Second, independent implementation of the receipt contract.

    A request key is bound on its first *successful* commit to that request's
    canonical identity.  A later call under the same key either replays the
    original result (identical identity) or conflicts (different identity);
    it never re-executes a prerequisite and never rewrites state.  A failed
    first attempt binds nothing.  Receipts are not removed by history
    compaction.
    """

    def __init__(self):
        self.head = 0
        self.latest = {}
        self.receipts = {}

    def plan(self, request_key, operations):
        if request_key is not None and request_key in self.receipts:
            prior = self.receipts[request_key]
            if prior["identity"] == request_identity(operations):
                return ("replay", prior)
            return ("conflict", None)
        return ("commit", None)

    def record_commit(self, request_key, operations):
        revision = self.head + 1
        self.head = revision
        for operation in operations:
            self.latest[(operation["collection"], operation["key"])] = revision
        if request_key is not None:
            self.receipts[request_key] = {
                "identity": request_identity(operations),
                "revision": revision,
                "count": len(operations),
            }
        return revision

    def current_revision(self, collection, key):
        return self.latest.get((collection, key), 0)


# ---------------------------------------------------------------------------
# Harness: executes requests and inspects the real on-disk database
# ---------------------------------------------------------------------------

TABLES = (
    "meta",
    "collections",
    "versions",
    "commits",
    "receipts",
    "index_defs",
    "index_values",
    "pins",
    "consumers",
)


class Harness:
    def __init__(self, path):
        self.path = str(path)
        self.db = Database(self.path)

    def close(self):
        self.db.close()

    def reopen(self):
        self.db = Database(self.path)
        return self

    def snapshot(self):
        """Exact rows of every persistent table, read outside the product API."""
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA query_only=ON")
            result = {}
            for table in TABLES:
                rows = connection.execute("SELECT * FROM " + table + " ORDER BY rowid").fetchall()
                result[table] = tuple(tuple(row) for row in rows)
            return result
        finally:
            connection.close()

    def receipts(self):
        return self.snapshot()["receipts"]

    def head(self):
        return int(self.db._store.get("head"))

    def floor(self):
        return int(self.db._store.get("floor"))

    def apply(self, request_key, operations):
        batch = self.db.batch(request_key) if request_key is not None else self.db.batch()
        for operation in operations:
            if operation["kind"] == "put":
                batch.put(
                    operation["collection"],
                    operation["key"],
                    operation["document"],
                    expected=operation["expected"],
                    expires_at=operation["expires_at"],
                )
            else:
                batch.delete(
                    operation["collection"],
                    operation["key"],
                    expected=operation["expected"],
                )
        return batch.commit()

    def verify_committed(self, model, operations, revision, request_key):
        """Check bodies, versions, expiry facts, the full commit and the receipt."""
        check(self.head() == revision, f"head is {self.head()}, expected {revision}")
        check(self.db.revision == revision, f"Database.revision is {self.db.revision}, expected {revision}")
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            for operation in operations:
                rows = connection.execute(
                    "SELECT * FROM versions WHERE collection=? AND key=? ORDER BY rev",
                    (operation["collection"], operation["key"]),
                ).fetchall()
                check(bool(rows), f"no version rows for {operation['collection']}/{operation['key']}")
                row = rows[-1]
                check(int(row["rev"]) == revision,
                      f"latest version of {operation['key']} is {row['rev']}, expected {revision}")
                if operation["kind"] == "delete":
                    check(row["body"] is None, f"delete of {operation['key']} did not store a tombstone")
                else:
                    check(row["body"] is not None, f"put of {operation['key']} stored a tombstone")
                    check(canon(json.loads(row["body"])) == canon(operation["document"]),
                          f"stored body of {operation['key']} differs from the accepted request")
                check(canon_expiry(row["expires"]) == canon_expiry(operation["expires_at"]),
                      f"stored expiry of {operation['key']} is {row['expires']}, expected {operation['expires_at']}")
                check(model.latest[(operation["collection"], operation["key"])] == revision,
                      "model and product disagree about the latest revision")

            commit_rows = connection.execute(
                "SELECT * FROM commits WHERE rev=?", (revision,)
            ).fetchall()
            check(len(commit_rows) == 1, f"expected exactly one full commit at revision {revision}")
            commit_row = commit_rows[0]
            payload = json.loads(commit_row["payload"])
            canonical_payload = canon(payload)
            canonical_request = canon(
                {"revision": revision, "operations": product_operations(operations)}
            )
            check(canonical_payload == canonical_request,
                  f"full commit payload at revision {revision} does not record the accepted operations")
            check(commit_row["request_key"] == request_key,
                  "commit row lost its request key")
        finally:
            connection.close()

    def verify_receipt(self, request_key, revision, count):
        if request_key is None:
            return
        connection = sqlite3.connect(self.path)
        try:
            row = connection.execute(
                "SELECT rev, count FROM receipts WHERE request_key=?", (request_key,)
            ).fetchone()
        finally:
            connection.close()
        check(row is not None, f"missing receipt for {request_key}")
        check(int(row[0]) == revision and int(row[1]) == count,
              f"receipt for {request_key} is {(row[0], row[1])}, expected {(revision, count)}")

    def assert_unchanged(self, before, context):
        after = self.snapshot()
        if after != before:
            changed = [table for table in TABLES if after[table] != before[table]]
            raise AssertionError(f"{context}: persistent state changed in {changed}")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# Timeline steps
# ---------------------------------------------------------------------------

STEPS = 0


def step(message):
    global STEPS
    STEPS += 1
    print(f"[{STEPS:03d}] {message}", file=sys.stderr)


def expect_commit(harness, model, label, request_key, operations, count=None):
    step(f"commit {label} (key={request_key!r})")
    verdict, _ = model.plan(request_key, operations)
    check(verdict == "commit", f"{label}: model expected {verdict}, not a fresh commit")
    before_head = harness.head()
    result = harness.apply(request_key, operations)
    revision = model.record_commit(request_key, operations)
    check(result.revision == revision,
          f"{label}: returned revision {result.revision}, model expected {revision}")
    check(result.count == (len(operations) if count is None else count),
          f"{label}: returned count {result.count}")
    check(result.replayed is False, f"{label}: fresh commit reported replayed=True")
    check(revision == before_head + 1, f"{label}: revision did not advance by exactly one")
    harness.verify_committed(model, operations, revision, request_key)
    harness.verify_receipt(request_key, revision, result.count)
    return result


def expect_replay(harness, model, label, request_key, operations):
    step(f"replay {label} (key={request_key!r})")
    verdict, prior = model.plan(request_key, operations)
    check(verdict == "replay", f"{label}: model expected {verdict}, not a replay")
    before = harness.snapshot()
    result = harness.apply(request_key, operations)
    check(result.replayed is True, f"{label}: replay reported replayed=False")
    check(result.revision == prior["revision"],
          f"{label}: replay returned revision {result.revision}, expected {prior['revision']}")
    check(result.count == prior["count"],
          f"{label}: replay returned count {result.count}, expected {prior['count']}")
    harness.assert_unchanged(before, f"{label}: replay")
    check(harness.head() == model.head, f"{label}: replay moved the head")
    return result


def expect_conflict(harness, model, label, request_key, operations):
    step(f"conflict {label} (key={request_key!r})")
    verdict, _ = model.plan(request_key, operations)
    check(verdict == "conflict", f"{label}: model expected {verdict}, not a conflict")
    before = harness.snapshot()
    try:
        harness.apply(request_key, operations)
    except Conflict as exc:
        message = str(exc)
    else:
        raise AssertionError(f"{label}: different request under the same key was accepted")
    harness.assert_unchanged(before, f"{label}: conflict")
    check(harness.head() == model.head, f"{label}: conflict moved the head")
    check("different operations" in message,
          f"{label}: conflict did not come from request identity: {message!r}")
    return message


def expect_failure(harness, model, label, request_key, operations, error):
    """A first attempt that must fail without binding a receipt."""
    step(f"failure {label} (key={request_key!r})")
    verdict, _ = model.plan(request_key, operations)
    check(verdict == "commit", f"{label}: model expected {verdict}, not a first attempt")
    before = harness.snapshot()
    try:
        harness.apply(request_key, operations)
    except error:
        pass
    else:
        raise AssertionError(f"{label}: expected {error.__name__}")
    harness.assert_unchanged(before, f"{label}: rejected request")
    # No receipt may exist for a request key that never committed.
    if request_key is not None:
        connection = sqlite3.connect(harness.path)
        try:
            row = connection.execute(
                "SELECT 1 FROM receipts WHERE request_key=?", (request_key,)
            ).fetchone()
        finally:
            connection.close()
        check(row is None, f"{label}: failed request occupied receipt {request_key!r}")
    return


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

def original_requests():
    a = {"value": 1, "labels": [1, 2]}
    b = {"value": 2}
    return [
        put_op(COLLECTION, "a", a, expected=0, expires_at=100),
        put_op(COLLECTION, "b", b, expected=0, expires_at=200),
    ]


def scenario(harness):
    model = TimelineModel()
    db = harness.db

    step("create loose docs collection")
    db.catalog.create(COLLECTION)

    # ---- Phase A: the exact request from the report -----------------------
    original = original_requests()
    first = expect_commit(harness, model, "original delivery request", "delivery", original)
    original_a = {"value": 1, "labels": [1, 2]}
    original_b = {"value": 2}

    expect_replay(harness, model, "identical delivery request", "delivery", original_requests())

    # JSON object member order is not part of request identity.
    reordered = [
        put_op(COLLECTION, "a", {"labels": [1, 2], "value": 1}, expected=0, expires_at=100),
        put_op(COLLECTION, "b", {"value": 2}, expected=0, expires_at=200),
    ]
    expect_replay(harness, model, "reordered document keys", "delivery", reordered)

    nested_first = [put_op(COLLECTION, "nested", {"outer": {"x": 1, "y": 2}}, expected=None)]
    nested_reordered = [put_op(COLLECTION, "nested", {"outer": {"y": 2, "x": 1}}, expected=None)]
    expect_commit(harness, model, "nested key order baseline", "nested_order", nested_first)
    expect_replay(harness, model, "reordered nested document keys", "nested_order", nested_reordered)

    # Distinct requests under the already-bound key must conflict, even though
    # a careless implementation could mistake each for equivalent work.
    different_expiry = [
        put_op(COLLECTION, "a", original_a, expected=0, expires_at=900),
        put_op(COLLECTION, "b", original_b, expected=0, expires_at=200),
    ]
    expect_conflict(harness, model, "expires_at 100 -> 900", "delivery", different_expiry)

    different_expected = [
        put_op(COLLECTION, "a", original_a, expected=None, expires_at=100),
        put_op(COLLECTION, "b", original_b, expected=0, expires_at=200),
    ]
    expect_conflict(harness, model, "expected 0 -> None", "delivery", different_expected)

    swapped = [
        put_op(COLLECTION, "b", original_b, expected=0, expires_at=200),
        put_op(COLLECTION, "a", original_a, expected=0, expires_at=100),
    ]
    expect_conflict(harness, model, "operation order swapped", "delivery", swapped)

    array_reordered = [
        put_op(COLLECTION, "a", {"value": 1, "labels": [2, 1]}, expected=0, expires_at=100),
        put_op(COLLECTION, "b", original_b, expected=0, expires_at=200),
    ]
    expect_conflict(harness, model, "array order changed", "delivery", array_reordered)

    explicit_null = [
        put_op(COLLECTION, "a", {"value": 1, "labels": [1, 2], "note": None},
               expected=0, expires_at=100),
        put_op(COLLECTION, "b", original_b, expected=0, expires_at=200),
    ]
    expect_conflict(harness, model, "missing member -> explicit null", "delivery", explicit_null)

    bool_for_int = [
        put_op(COLLECTION, "a", {"value": True, "labels": [1, 2]}, expected=0, expires_at=100),
        put_op(COLLECTION, "b", original_b, expected=0, expires_at=200),
    ]
    expect_conflict(harness, model, "nested true for 1", "delivery", bool_for_int)

    # ---- Phase A2: replayed receipt survives later writes -----------------
    step("change a through a different request")
    current_a = db.get(COLLECTION, "a")
    model_a_revision = model.current_revision(COLLECTION, "a")
    check(current_a is not None and current_a.revision == model_a_revision,
          "product and model disagree about a's revision before the mutation")
    mutate = [put_op(COLLECTION, "a", {"value": 3}, expected=current_a.revision)]
    mutated = expect_commit(harness, model, "mutate a to value 3", "mutate_a", mutate)
    check(db.get(COLLECTION, "a").document == {"value": 3}, "a was not changed to value 3")

    # The original request must still return the first receipt, even though its
    # expected revision is stale and the current body is different.  It must not
    # restore the old body.
    replayed = expect_replay(harness, model, "original request after a changed", "delivery", original_requests())
    check(replayed.revision == first.revision, "original replay lost its original revision")
    check(db.get(COLLECTION, "a").document == {"value": 3}, "original replay restored a stale body")

    # Changing the request body to the *current* value does not make it a legal
    # retry of the old key: request identity is not "does the body now match".
    current_body = [
        put_op(COLLECTION, "a", {"value": 3}, expected=None, expires_at=100),
        put_op(COLLECTION, "b", original_b, expected=None, expires_at=200),
    ]
    expect_conflict(harness, model, "body rewritten to current value", "delivery", current_body)
    check(db.get(COLLECTION, "a").document == {"value": 3}, "conflicting retry rewrote a")

    # ---- Phase B: delete and same-key rebuild -----------------------------
    # Read the current version through the product's own read path, then derive
    # the independent model value from it; both must agree before proceeding.
    current_a = db.get(COLLECTION, "a")
    model_a_revision = model.current_revision(COLLECTION, "a")
    check(current_a is not None and current_a.revision == model_a_revision,
          f"a is at revision {None if current_a is None else current_a.revision}, model says {model_a_revision}")
    a_revision = current_a.revision

    first_delete_life2 = [delete_op(COLLECTION, "a", expected=a_revision)]
    deleted = expect_commit(harness, model, "delete a in its second life", "del_life2", first_delete_life2)
    check(db.get(COLLECTION, "a") is None, "delete did not hide a")

    expect_replay(harness, model, "repeat delete of second life", "del_life2", first_delete_life2)
    expect_conflict(harness, model, "delete with a different expected revision", "del_life2",
                    [delete_op(COLLECTION, "a", expected=a_revision + 1)])

    rebuilt = expect_commit(harness, model, "rebuild a in a third life", "rebuild_life2",
                            [put_op(COLLECTION, "a", {"value": 3}, expected=a_revision + 1)])
    check(db.get(COLLECTION, "a").document == {"value": 3}, "rebuild did not restore a")

    # The old delete receipt still describes the first deletion; it neither
    # replays the stale body nor pretends to delete the new life.
    old_delete_replay = expect_replay(harness, model, "old delete after rebuild", "del_life2",
                                      first_delete_life2)
    check(old_delete_replay.revision == deleted.revision,
          "old delete replay returned a different revision")
    check(db.get(COLLECTION, "a").document == {"value": 3},
          "old delete replay deleted the rebuilt document")

    rebuilt_revision = db.get(COLLECTION, "a").revision
    check(rebuilt_revision == model.current_revision(COLLECTION, "a"),
          "model and product disagree about the rebuilt revision")
    # Same key, same collection, same intent (delete), but a different life:
    # only the target revision differs, and this request must not impersonate
    # the earlier deletion.
    expect_conflict(harness, model, "delete request from a later life", "del_life2",
                    [delete_op(COLLECTION, "a", expected=rebuilt_revision)])
    expect_commit(harness, model, "delete from a later life under a new key", "del_life3",
                  [delete_op(COLLECTION, "a", expected=rebuilt_revision)])
    expect_replay(harness, model, "replay third-life delete", "del_life3",
                  [delete_op(COLLECTION, "a", expected=rebuilt_revision)])
    expect_replay(harness, model, "second-life delete after third-life tombstone", "del_life2",
                  first_delete_life2)

    # Tombstone versions are still revisions: a delete can target a tombstone,
    # and a put under that same key is a different request.
    tombstone_revision = model.current_revision(COLLECTION, "a")
    kind_delete = [delete_op(COLLECTION, "a", expected=tombstone_revision)]
    expect_commit(harness, model, "delete a tombstone", "kind_probe", kind_delete)
    expect_conflict(harness, model, "put under a key first bound to a delete", "kind_probe",
                    [put_op(COLLECTION, "a", {"value": 9}, expected=tombstone_revision + 1)])
    check(db.get(COLLECTION, "a") is None, "conflicting put resurrected a")

    # Type and shape probes where a fresh execution would otherwise succeed
    # (expected=None), so a wrong "re-execute" verdict cannot hide behind a
    # version conflict.
    null_probe = [put_op(COLLECTION, "c", {"k": None}, expected=None)]
    expect_commit(harness, model, "explicit null baseline", "null_missing", null_probe)
    expect_conflict(harness, model, "explicit null vs absent member", "null_missing",
                    [put_op(COLLECTION, "c", {}, expected=None)])

    bool_probe = [put_op(COLLECTION, "d", {"flag": True}, expected=None)]
    expect_commit(harness, model, "true baseline", "bool_int", bool_probe)
    expect_conflict(harness, model, "true vs 1 in body", "bool_int",
                    [put_op(COLLECTION, "d", {"flag": 1}, expected=None)])

    array_probe = [put_op(COLLECTION, "e", {"a": [1, 2]}, expected=None)]
    expect_commit(harness, model, "ordered array baseline", "array_order", array_probe)
    expect_conflict(harness, model, "array [1,2] vs [2,1]", "array_order",
                    [put_op(COLLECTION, "e", {"a": [2, 1]}, expected=None)])

    # Collection and document key are part of the request identity: a request
    # for another key or collection is a different request even when the body,
    # expected revision and expiry are byte-for-byte the same.
    key_probe = [put_op(COLLECTION, "g", {"value": 7}, expected=None, expires_at=300)]
    expect_commit(harness, model, "document key baseline", "key_identity", key_probe)
    expect_conflict(harness, model, "same body under a different document key", "key_identity",
                    [put_op(COLLECTION, "h", {"value": 7}, expected=None, expires_at=300)])
    check(db.get(COLLECTION, "h") is None, "conflicting key request created h")

    collection_probe = [put_op(COLLECTION, "i", {"value": 8}, expected=None, expires_at=300)]
    expect_commit(harness, model, "collection baseline", "collection_identity", collection_probe)
    db.catalog.create("delivery_copy")
    expect_conflict(harness, model, "same body under a different collection", "collection_identity",
                    [put_op("delivery_copy", "i", {"value": 8}, expected=None, expires_at=300)])
    check(db.get("delivery_copy", "i") is None, "conflicting collection request created a document")

    # ---- Phase C: genuine rejections and receipt hygiene ------------------
    stale_expected = 0
    step("version precondition rejection")
    a_now = model.current_revision(COLLECTION, "a")
    expect_failure(harness, model, "stale expected revision", "fail_version",
                   [put_op(COLLECTION, "a", {"value": 99}, expected=stale_expected)], Conflict)
    expect_commit(harness, model, "same key after the failed attempt, corrected", "fail_version",
                  [put_op(COLLECTION, "a", {"value": 99}, expected=a_now)])

    step("create strict collection for schema rejection")
    db.catalog.create("strict", {"fields": {"n": "integer"}, "required": ["n"], "additional": False})
    expect_failure(harness, model, "boolean rejected where integer is required", "fail_schema",
                   [put_op("strict", "x", {"n": True})], InvalidDocument)
    expect_commit(harness, model, "same key after schema rejection, valid", "fail_schema",
                  [put_op("strict", "x", {"n": 1})])

    step("create unique index and exercise a real unique rejection")
    db.indexes.create(COLLECTION, "unique_value", "value", unique=True)
    expect_failure(harness, model, "duplicate unique value", "fail_unique",
                   [put_op(COLLECTION, "f", {"value": 2})], Conflict)
    expect_commit(harness, model, "same key after unique rejection, valid", "fail_unique",
                  [put_op(COLLECTION, "f", {"value": 10})])

    step("writes without a request key are not deduplicated")
    head_before = harness.head()
    no_key_one = db.batch().put(COLLECTION, "b", {"value": 20}, expected=None).commit()
    check(no_key_one.replayed is False, "keyless write reported a replay")
    no_key_two = db.batch().put(COLLECTION, "b", {"value": 20}, expected=None).commit()
    check(no_key_two.replayed is False, "keyless repeat write reported a replay")
    check(no_key_two.revision == no_key_one.revision + 1,
          "keyless repeat write did not commit a new revision")
    check(harness.head() == head_before + 2, "keyless writes did not advance the head twice")
    connection = sqlite3.connect(harness.path)
    try:
        keyless = connection.execute(
            "SELECT request_key FROM commits WHERE rev IN (?,?)", (no_key_one.revision, no_key_two.revision)
        ).fetchall()
    finally:
        connection.close()
    check(all(row[0] is None for row in keyless), "keyless commit stored a request key")
    model.record_commit(None, [put_op(COLLECTION, "b", {"value": 20})])
    model.record_commit(None, [put_op(COLLECTION, "b", {"value": 20})])
    # The model's head is now aligned; verify the two bodies directly.
    check(db.get(COLLECTION, "b").document == {"value": 20}, "keyless write body is wrong")

    step("replica rejects a primary write and binds no receipt")
    backup_path = Path(harness.path).with_name("receipt-backup.db")
    replica_path = Path(harness.path).with_name("receipt-replica.db")
    backup = db.archives.backup(backup_path)
    Archives.restore(backup_path, replica_path, backup["sha256"], follower=True)
    with Database(replica_path) as replica:
        replica_head = replica.revision
        before = _replica_snapshot(replica_path)
        try:
            replica.batch("replica_probe").put(COLLECTION, "z", {"value": 1}).commit()
        except ReadOnly:
            pass
        else:
            raise AssertionError("replica accepted a primary write")
        check(replica.revision == replica_head, "replica write attempt moved its head")
        check(_replica_snapshot(replica_path) == before,
              "replica write attempt changed persistent state")
        connection = sqlite3.connect(str(replica_path))
        try:
            check(connection.execute(
                "SELECT 1 FROM receipts WHERE request_key='replica_probe'").fetchone() is None,
                "replica write attempt occupied a receipt")
        finally:
            connection.close()

    # ---- Phase D: compaction, close, reopen ------------------------------
    step("compact all history and verify receipts survive")
    receipts_before = harness.receipts()
    expected_keys = {key for key in model.receipts}
    connection = sqlite3.connect(harness.path)
    try:
        commits_before = connection.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
        versions_before = connection.execute("SELECT COUNT(*) FROM versions").fetchone()[0]
    finally:
        connection.close()
    check(commits_before > 0, "timeline produced no commits to compact")
    compacted = db.retention.compact(db.revision)
    check(harness.floor() == compacted["floor"], "compaction floor mismatch")
    check(compacted["removed_commits"] > 0,
          "compaction retained every commit, so replay could cheat with full history")
    check(compacted["removed_versions"] > 0,
          "compaction retained every version, so replay could cheat with full history")
    connection = sqlite3.connect(harness.path)
    try:
        commits_after = connection.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
        versions_after = connection.execute("SELECT COUNT(*) FROM versions").fetchone()[0]
    finally:
        connection.close()
    check(commits_after == 0, f"compaction left {commits_after} full commits")
    check(versions_after < versions_before, "compaction did not remove superseded versions")
    check(harness.receipts() == receipts_before, "compaction rewrote or cleared receipts")
    check(len(receipts_before) == len(expected_keys),
          f"receipt count {len(receipts_before)} differs from model {len(expected_keys)}")
    step(f"receipts retained through compaction: {len(receipts_before)}")

    step("close and reopen the database")
    harness.close()
    harness.reopen()
    db = harness.db
    check(harness.receipts() == receipts_before, "reopen changed receipts")
    check(int(db._store.get("floor")) > 0, "reopen lost the advanced floor")

    # Old revisions are no longer readable from history, yet the legal retry
    # must still return the original receipt.
    step("old revision is genuinely gone from history")
    try:
        db.get(COLLECTION, "a", first.revision)
    except SnapshotTooOld:
        pass
    else:
        raise AssertionError("old revision remained readable; test did not prove receipt-only replay")

    fingerprint_before = db.diagnostics.fingerprint()
    original_replay = expect_replay(harness, model, "original request after compaction and reopen",
                                    "delivery", original_requests())
    check(original_replay.revision == first.revision, "post-reopen replay lost the original revision")
    check(db.diagnostics.fingerprint() == fingerprint_before,
          "replay after reopen changed the current fingerprint")
    check(db.diagnostics.inspect(deep=True)["ok"], "database is not healthy after replay")

    expect_conflict(harness, model, "different expires after compaction", "delivery", different_expiry)
    expect_conflict(harness, model, "swapped order after compaction", "delivery", swapped)
    expect_conflict(harness, model, "body rewritten to current value after compaction", "delivery",
                    current_body)

    expect_replay(harness, model, "second-life delete after compaction", "del_life2", first_delete_life2)
    expect_replay(harness, model, "third-life delete after compaction", "del_life3",
                  [delete_op(COLLECTION, "a", expected=rebuilt_revision)])

    check(harness.receipts() == receipts_before,
          "replays or conflicts rewrote receipts after compaction")
    check({row[0] for row in harness.receipts()} == expected_keys,
          "request keys changed during replay/conflict handling")

    step("a genuinely new request still commits after compaction and reopen")
    expect_commit(harness, model, "new request after reopen", "fresh_after_reopen",
                  [put_op(COLLECTION, "fresh", {"value": 123})])
    check(db.diagnostics.inspect(deep=True)["ok"], "database is not healthy after a fresh commit")

    step("all independent expectations matched")
    return {
        "steps": STEPS,
        "receipts": len(receipts_before),
        "compacted": compacted,
        "head": harness.head(),
        "floor": harness.floor(),
        "original_revision": first.revision,
    }


def _replica_snapshot(path):
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("PRAGMA query_only=ON")
        return {
            table: tuple(tuple(row) for row in connection.execute(
                "SELECT * FROM " + table + " ORDER BY rowid").fetchall())
            for table in TABLES
        }
    finally:
        connection.close()


def main():
    with tempfile.TemporaryDirectory(prefix="strataleaf-receipt-regression-") as directory:
        path = Path(directory) / "receipts.db"
        harness = Harness(path)
        try:
            summary = scenario(harness)
        except BaseException:
            harness.close()
            traceback.print_exc()
            print(json.dumps({"ok": False, "steps_completed": STEPS}, sort_keys=True))
            return 1
        else:
            harness.close()
            print(json.dumps({"ok": True, **summary}, sort_keys=True))
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
