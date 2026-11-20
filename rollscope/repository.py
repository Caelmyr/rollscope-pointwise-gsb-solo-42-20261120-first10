import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from .capture_codec import canonical_json, capture_bytes, capture_hash, read_capture
from .domain import MissingRecord, RevisionConflict, UserInputError, clean_text
from .grain import GrainSelection
from .records import MAX_SNAPSHOTS, CaptureState, InspectionSnapshot, identifier, utc_now


SCHEMA_VERSION = 1
MAX_CAPTURES = 64
MAX_CAPTURE_BYTES = 512 * 1024 * 1024
STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS captures (
    id TEXT PRIMARY KEY,
    source_sha TEXT NOT NULL,
    original_name TEXT,
    original_sha TEXT,
    bundle BLOB NOT NULL,
    overview TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id TEXT PRIMARY KEY,
    capture_id TEXT NOT NULL REFERENCES captures(id),
    created_at TEXT NOT NULL,
    record TEXT NOT NULL,
    summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grain_links (
    capture_id TEXT PRIMARY KEY REFERENCES captures(id),
    selection TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS snapshots_by_capture ON snapshots(capture_id, created_at DESC, id);
CREATE INDEX IF NOT EXISTS captures_by_date ON captures(created_at DESC, id);
"""


def read_json(value):
    return json.loads(value)


def stored_json(value):
    return canonical_json(value).decode("utf-8")


@dataclass(frozen=True)
class CaptureRecord:
    id: str
    sha256: str
    original_name: str | None
    original_sha256: str | None
    created_at: str
    capture: object
    state: CaptureState

    def manifest(self):
        return {
            "id": self.id,
            "sha256": self.sha256,
            "name": self.capture.metadata.name,
            "metadata": self.capture.metadata.as_dict(),
            "source_name": self.original_name,
            "source_sha256": self.original_sha256,
            "created_at": self.created_at,
            "overview": self.capture.overview(),
        }


class Repository:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.database = directory / "rollscope.sqlite3"
        self.initialize()

    @contextmanager
    def open(self):
        with closing(sqlite3.connect(self.database, timeout=10)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection

    @contextmanager
    def writing(self):
        with self.open() as connection:
            connection.execute("BEGIN IMMEDIATE")
            with connection:
                yield connection

    def initialize(self):
        with self.open() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise RuntimeError(f"Unsupported RollScope database version {version}.")
            connection.executescript(STATE_SCHEMA)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def health(self):
        with self.open() as connection:
            connection.execute("SELECT id FROM captures LIMIT 1").fetchone()

    def add_capture(self, capture, original_name=None, original_payload=None):
        ref = identifier()
        source_sha = capture_hash(capture)
        original_sha = None
        if original_name is not None:
            original_name = clean_text(Path(original_name).name, "Original filename", 180)
        if original_payload is not None:
            original_sha = hashlib.sha256(original_payload).hexdigest()
        now = utc_now()
        record = CaptureRecord(
            ref, source_sha, original_name, original_sha, now, capture, CaptureState.create(now)
        )
        with self.writing() as connection:
            self.insert_capture(connection, record)
        return record

    def insert_capture(self, connection, record):
        capture = record.capture
        bundle = capture_bytes(capture)
        listing = capture.metadata.as_dict() | {"overview": capture.overview()}
        totals = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(length(bundle)), 0) FROM captures"
        ).fetchone()
        if totals[0] >= MAX_CAPTURES:
            raise UserInputError("The local library is full (64 captures).")
        if totals[1] + len(bundle) > MAX_CAPTURE_BYTES:
            raise UserInputError("The local library has reached its 512 MiB source limit.")
        connection.execute(
            """
            INSERT INTO captures(
                id, source_sha, original_name, original_sha, bundle,
                overview, state, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.sha256,
                record.original_name,
                record.original_sha256,
                bundle,
                stored_json(listing),
                stored_json(record.state.as_dict()),
                record.created_at,
                record.state.updated_at,
            ),
        )

    def restore(self, project):
        capture = project.capture
        record = CaptureRecord(
            identifier(),
            capture_hash(capture),
            project.source_name,
            project.source_sha,
            utc_now(),
            capture,
            project.state,
        )
        with self.writing() as connection:
            self.insert_capture(connection, record)
            for snapshot in project.snapshots:
                self.insert_snapshot(connection, record.id, replace(snapshot, id=identifier()))
        return record

    def project_records(self, ref):
        with self.open() as connection:
            connection.execute("BEGIN")
            record = self.read_capture_row(connection, ref)
            rows = connection.execute(
                """
                SELECT record FROM snapshots
                WHERE capture_id = ? ORDER BY created_at DESC, id
                """,
                (ref,),
            ).fetchall()
            snapshots = [
                InspectionSnapshot.stored(read_json(row["record"])) for row in rows
            ]
        return record, snapshots

    def list_captures(self):
        with self.open() as connection:
            rows = connection.execute(
                """
                SELECT id, source_sha, overview, created_at, updated_at
                FROM captures ORDER BY created_at DESC, id
                """
            ).fetchall()
        return [
            read_json(row["overview"])
            | {
                "id": row["id"],
                "sha256": row["source_sha"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def get_capture(self, ref):
        with self.open() as connection:
            return self.read_capture_row(connection, ref)

    def read_capture_row(self, connection, ref):
        row = connection.execute("SELECT * FROM captures WHERE id = ?", (ref,)).fetchone()
        if row is None:
            raise MissingRecord("That capture is not in the local library.")
        return self.capture_record(row)

    def capture_record(self, row):
        return CaptureRecord(
            row["id"],
            row["source_sha"],
            row["original_name"],
            row["original_sha"],
            row["created_at"],
            read_capture(row["bundle"]),
            CaptureState.from_dict(read_json(row["state"])),
        )

    def get_state(self, ref):
        with self.open() as connection:
            return self.read_state(connection, ref)

    def read_state(self, connection, ref):
        row = connection.execute("SELECT state FROM captures WHERE id = ?", (ref,)).fetchone()
        if row is None:
            raise MissingRecord("That capture is not in the local library.")
        return CaptureState.from_dict(read_json(row["state"]))

    def require_revision(self, connection, ref, expected):
        state = self.read_state(connection, ref)
        if state.revision != expected:
            raise RevisionConflict(state.revision)
        return state

    def write_state(self, connection, ref, state):
        connection.execute(
            "UPDATE captures SET state = ?, updated_at = ? WHERE id = ?",
            (stored_json(state.as_dict()), state.updated_at, ref),
        )

    def mutate(self, ref, expected, transform):
        with self.writing() as connection:
            current = self.require_revision(connection, ref, expected)
            candidate = transform(current)
            modified = candidate.next_revision(utc_now())
            self.write_state(connection, ref, modified)
        return modified

    def save_snapshot(self, ref, expected, snapshot):
        now = snapshot.timestamp
        with self.writing() as connection:
            state = self.require_revision(connection, ref, expected)
            count = connection.execute(
                "SELECT COUNT(*) FROM snapshots WHERE capture_id = ?", (ref,)
            ).fetchone()[0]
            if count >= MAX_SNAPSHOTS:
                raise UserInputError(f"One capture can hold at most {MAX_SNAPSHOTS} snapshots.")
            self.insert_snapshot(connection, ref, snapshot)
            updated = state.next_revision(now)
            self.write_state(connection, ref, updated)
        return updated

    def insert_snapshot(self, connection, ref, snapshot):
        connection.execute(
            "INSERT INTO snapshots(id, capture_id, created_at, record, summary) "
            "VALUES(?, ?, ?, ?, ?)",
            (
                snapshot.id,
                ref,
                snapshot.timestamp,
                stored_json(snapshot.as_dict()),
                stored_json(snapshot.summary()),
            ),
        )

    def list_snapshots(self, ref):
        with self.open() as connection:
            rows = connection.execute(
                """
                SELECT summary FROM snapshots
                WHERE capture_id = ?
                ORDER BY created_at DESC, id
                """,
                (ref,),
            ).fetchall()
        return [read_json(row["summary"]) for row in rows]

    def get_snapshot(self, ref, snapshot):
        with self.open() as connection:
            row = connection.execute(
                "SELECT record FROM snapshots WHERE capture_id = ? AND id = ?",
                (ref, snapshot),
            ).fetchone()
        if row is None:
            raise MissingRecord("That inspection snapshot is no longer in the local library.")
        return InspectionSnapshot.stored(read_json(row["record"]))

    def source_bundle(self, ref):
        with self.open() as connection:
            row = connection.execute(
                "SELECT bundle FROM captures WHERE id = ?", (ref,)
            ).fetchone()
        if row is None:
            raise MissingRecord("That capture is not in the local library.")
        return row["bundle"]

    def grain_selection(self, ref):
        with self.open() as connection:
            row = connection.execute(
                "SELECT selection FROM grain_links WHERE capture_id = ?", (ref,)
            ).fetchone()
        if row is None:
            return None
        return GrainSelection.parse(read_json(row["selection"]))

    def write_grain_selection(self, connection, ref, selection):
        connection.execute(
            "INSERT INTO grain_links(capture_id, selection, updated_at) VALUES(?, ?, ?) "
            "ON CONFLICT(capture_id) DO UPDATE SET selection = excluded.selection, "
            "updated_at = excluded.updated_at",
            (ref, stored_json(selection.as_dict()), utc_now()),
        )

    def set_grain_selection(self, ref, selection):
        with self.writing() as connection:
            if connection.execute(
                "SELECT id FROM captures WHERE id = ?", (ref,)
            ).fetchone() is None:
                raise MissingRecord("That capture is not in the local library.")
            self.write_grain_selection(connection, ref, selection)
        return selection

    def with_grain_selections(self, refs):
        ids = list(dict.fromkeys(refs))
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        with self.open() as connection:
            rows = connection.execute(
                f"SELECT capture_id, selection FROM grain_links WHERE capture_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {
            row["capture_id"]: GrainSelection.parse(read_json(row["selection"]))
            for row in rows
        }

    def grain_linked_captures(self):
        with self.open() as connection:
            rows = connection.execute(
                "SELECT capture_id, selection FROM grain_links ORDER BY updated_at DESC"
            ).fetchall()
        return [
            read_json(row["selection"]) | {"capture_id": row["capture_id"]}
            for row in rows
        ]

    def list_window_catalog(self):
        """Every saved window in the library, for choosing grain references across captures.

        Only the stored state and header are read; capture bundles are left on disk.
        """
        with self.open() as connection:
            rows = connection.execute(
                "SELECT id, overview, state FROM captures ORDER BY created_at DESC, id"
            ).fetchall()
        catalog = []
        for row in rows:
            overview = read_json(row["overview"])
            state = CaptureState.from_dict(read_json(row["state"]))
            catalog.append(
                {
                    "capture_ref": row["id"],
                    "capture_name": overview.get("name", ""),
                    "windows": state.windows,
                }
            )
        return catalog
