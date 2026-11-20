import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from zipfile import BadZipFile, ZipFile

from .analysis import SECTION_TITLES
from .capture_codec import MAX_BUNDLE_BYTES, capture_bytes, capture_hash, read_capture
from .domain import UserInputError, clean_text, exact_keys, expect_object, reject_unknown, whole_number
from .evidence import EVIDENCE_FORMAT, Evidence
from .records import MAX_SNAPSHOTS, MAX_WINDOWS, CaptureState, InspectionSnapshot, ReviewWindow
from .settings import AnalysisSettings
from .transverse import TransverseCalibration


MAX_PROJECT_BYTES = 48 * 1024 * 1024
MAX_ARCHIVED_REPORT_BYTES = 24 * 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}")
KEY = re.compile(r"[0-9a-f]{16}")
ARCHIVE_ENTRIES = {
    "capture.npz": MAX_BUNDLE_BYTES,
    "project.json": 16 * 1024,
    "state.json": 256 * 1024,
    "snapshots.json": MAX_ARCHIVED_REPORT_BYTES,
}


@dataclass(frozen=True)
class RestoredProject:
    capture: object
    source_sha: str | None
    source_name: str | None
    state: CaptureState
    snapshots: tuple


def project_export(record, snapshots):
    bundle = Evidence("project")
    bundle.add("capture.npz", capture_bytes(record.capture))
    manifest = record.manifest()
    bundle.json("project.json", manifest)
    bundle.json("state.json", record.state.as_dict())
    bundle.json("snapshots.json", [snapshot.as_dict() for snapshot in snapshots])
    return bundle.make({"capture_sha256": record.sha256, "source_revision": record.state.revision})


def read_project(payload):
    if len(payload) > MAX_PROJECT_BYTES:
        raise UserInputError("The project archive exceeds 48 MiB.")
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            members, manifest = archive_members(archive)
    except (BadZipFile, OSError, RuntimeError, EOFError) as error:
        raise UserInputError("The project could not be read as a RollScope archive.") from error
    capture = read_capture(members["capture.npz"])
    metadata = json_document(members["project.json"], "Project header")
    source_hash = capture_hash(capture)
    binding = expect_object(manifest.get("binding"), "Project binding")
    if binding.get("capture_sha256") != source_hash or metadata.get("sha256") != source_hash:
        raise UserInputError("The capture contents do not match the project archive.")
    state = restore_state(members["state.json"], capture)
    if binding.get("source_revision") != state.revision:
        raise UserInputError("The project state does not match the exported revision.")
    snapshots = restore_snapshots(members["snapshots.json"], state, source_hash)
    original_sha = read_hash(metadata.get("source_sha256"), optional=True)
    source_name = metadata.get("source_name")
    if source_name is not None:
        source_name = clean_text(source_name, "Archived source filename", 180)
    return RestoredProject(capture, original_sha, source_name, state, snapshots)


def bounded_read(archive, name, limit):
    member = archive.getinfo(name)
    if member.file_size > limit:
        raise UserInputError(f"The project member {name} is too large.")
    with archive.open(member) as stream:
        payload = stream.read(limit + 1)
    if len(payload) != member.file_size:
        raise UserInputError(f"The project member {name} has an invalid size.")
    return payload


def archive_members(archive):
    expected = sorted((*ARCHIVE_ENTRIES, "manifest.json"))
    if sorted(archive.namelist()) != expected:
        raise UserInputError("Select a complete RollScope project archive.")
    document = bounded_read(archive, "manifest.json", 32 * 1024)
    manifest = json_document(document, "Archive manifest")
    if manifest.get("format") != EVIDENCE_FORMAT or manifest.get("kind") != "project":
        raise UserInputError("This archive does not contain a RollScope project.")
    declarations = manifest.get("members")
    if not isinstance(declarations, list) or len(declarations) != len(ARCHIVE_ENTRIES):
        raise UserInputError("The project archive member list is invalid.")
    found = {}
    for entry in declarations:
        entry = expect_object(entry, "Project member")
        exact_keys(entry, ("path", "sha256", "bytes"), "Project member")
        name = entry["path"]
        if not isinstance(name, str) or name in found or name not in ARCHIVE_ENTRIES:
            raise UserInputError("The project archive lists an unexpected or repeated member.")
        expected_bytes = whole_number(entry["bytes"], "Member size", 0, ARCHIVE_ENTRIES[name])
        checksum = read_hash(entry["sha256"])
        data = bounded_read(archive, name, ARCHIVE_ENTRIES[name])
        if len(data) != expected_bytes or hashlib.sha256(data).hexdigest() != checksum:
            raise UserInputError(f"The checksum for {name} does not match the manifest.")
        found[name] = data
    return found, manifest


def reject_constant(value):
    raise ValueError(value)


def json_document(payload, label, expected=dict):
    try:
        data = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise UserInputError(f"{label} must be valid UTF-8 JSON.") from error
    if not isinstance(data, expected):
        raise UserInputError(f"{label} has an invalid structure.")
    return data


def read_hash(value, optional=False):
    if value is None and optional:
        return value
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise UserInputError("An archived checksum is invalid.")
    return value


def read_key(value):
    if not isinstance(value, str) or KEY.fullmatch(value) is None:
        raise UserInputError("An archived record ID is invalid.")
    return value


def timestamp(value):
    if not isinstance(value, str) or len(value) > 50:
        raise UserInputError("An archive timestamp is invalid.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise UserInputError("An archive timestamp is invalid.") from error
    if parsed.tzinfo is None:
        raise UserInputError("Archive timestamps must include a time zone.")
    return value


def restore_state(payload, capture):
    data = json_document(payload, "Project state")
    expected = (
        "revision",
        "notes",
        "settings",
        "windows",
        "updated_at",
        "transverse_calibration",
    )
    # Archives written before transverse calibration existed omit that key.
    reject_unknown(data, expected, "Project state")
    for key in ("revision", "notes", "settings", "windows", "updated_at"):
        if key not in data:
            raise UserInputError(f"Project state is missing a required field: {key}.")
    windows = data["windows"]
    if not isinstance(windows, list) or len(windows) > MAX_WINDOWS:
        raise UserInputError("The archive exceeds the review window limit.")
    records = []
    ids = set()
    for raw in windows:
        entry = expect_object(raw, "Archived review window").copy()
        ref = read_key(entry.pop("id", None))
        if ref in ids:
            raise UserInputError("The archive contains duplicate review window IDs.")
        ids.add(ref)
        records.append(ReviewWindow.from_request(entry, capture, ref))
    calibration_data = data.get("transverse_calibration")
    transverse_calibration = (
        TransverseCalibration.parse(calibration_data) if calibration_data else None
    )
    if transverse_calibration is not None and transverse_calibration.columns != capture.columns:
        raise UserInputError(
            "The archived transverse calibration was built for a different column count."
        )
    return CaptureState(
        whole_number(data["revision"], "Archived revision", 0, 2**31 - 1),
        AnalysisSettings.parse(data["settings"]),
        tuple(records),
        clean_text(data["notes"], "Archived capture notes", 4000, True),
        timestamp(data["updated_at"]),
        transverse_calibration,
    )


def restore_snapshots(payload, state, source_hash):
    data = json_document(payload, "Inspection snapshots", expected=list)
    if len(data) > MAX_SNAPSHOTS:
        raise UserInputError("The archive exceeds the inspection snapshot limit.")
    records = []
    ids = set()
    for raw in data:
        snapshot = restore_snapshot(raw, state, source_hash)
        if snapshot.id in ids:
            raise UserInputError("The project archive contains duplicate snapshot IDs.")
        ids.add(snapshot.id)
        records.append(snapshot)
    return tuple(records)


def restore_snapshot(data, state, source_hash):
    entry = expect_object(data, "Archived inspection snapshot")
    required = (
        "id", "label", "section", "settings", "result", "capture_sha",
        "source_revision", "version", "timestamp",
    )
    exact_keys(entry, required, "Archived inspection snapshot")
    if entry["capture_sha"] != source_hash:
        raise UserInputError("An inspection snapshot belongs to a different capture.")
    section = entry["section"]
    if not isinstance(section, str) or section not in SECTION_TITLES:
        raise UserInputError("An inspection snapshot reports an unsupported section.")
    result = expect_object(entry["result"], "Archived inspection result")
    validate_result(result, section)
    return InspectionSnapshot(
        read_key(entry["id"]),
        clean_text(entry["label"], "Archived snapshot name", 75),
        section,
        AnalysisSettings.parse(entry["settings"]),
        result,
        source_hash,
        whole_number(entry["source_revision"], "Source revision", 0, state.revision),
        clean_text(entry["version"], "Engine version", 40),
        timestamp(entry["timestamp"]),
    )


def validate_result(result, section):
    exact_keys(result, ("section", "name", "summary", "tables", "charts"), "Archived result")
    if result["section"] != section or result["name"] != SECTION_TITLES[section]:
        raise UserInputError("The archived result does not match its analysis section.")
    expect_object(result["summary"], "Archived result summary")
    validate_datum(result["summary"], 0)
    sequences = (result["charts"], result["tables"])
    if any(not isinstance(items, list) or len(items) > 12 for items in sequences):
        raise UserInputError("The archive contains an invalid result table or chart list.")
    for entry in result["tables"]:
        item = expect_object(entry, "Archived result table")
        exact_keys(item, ("key", "title", "rows"), "Archived result table")
        validate_title(item["title"])
        key = item["key"]
        if not isinstance(key, str) or not re.fullmatch("[a-z0-9_]{1,36}", key):
            raise UserInputError("The archive contains an invalid result key.")
        validate_rows(item["rows"])
    for entry in result["charts"]:
        item = expect_object(entry, "Archived result chart")
        exact_keys(item, ("title", "x", "y", "unit", "records"), "Archived result chart")
        validate_title(item["title"])
        x = item["x"]
        if not isinstance(x, str) or not re.fullmatch("[a-z0-9_]{1,36}", x):
            raise UserInputError("The archive contains an invalid plot axis.")
        clean_text(item["unit"], "Measure unit", 60)
        series = item["y"]
        if not isinstance(series, list) or not 1 <= len(series) <= 4:
            raise UserInputError("The archive contains an invalid plot series.")
        for metric in series:
            clean_text(metric, "Measure name", 60)
        validate_rows(item["records"])


def validate_title(title):
    clean_text(title, "Result title", 100)


def validate_rows(rows):
    if not isinstance(rows, list) or len(rows) > 30000:
        raise UserInputError("An archived result has too many table rows.")
    for row in rows:
        validate_datum(expect_object(row, "Archived result row"), 0)


def validate_datum(value, depth):
    if depth > 5:
        raise UserInputError("The archived report contains an unsupported nested structure.")
    if isinstance(value, dict):
        if len(value) > 40:
            raise UserInputError("The archived report contains too many fields.")
        for key, child in value.items():
            clean_text(key, "Result field", 80)
            validate_datum(child, depth + 1)
    elif isinstance(value, list):
        if len(value) > 20:
            raise UserInputError("The archived report contains an oversized result cell.")
        for datum in value:
            validate_datum(datum, depth + 1)
    elif isinstance(value, str):
        clean_text(value, "Result value", 400, True)
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise UserInputError("The archived report contains an unsupported result value.")
    elif isinstance(value, float) and not math.isfinite(value):
        raise UserInputError("The archived report contains a nonfinite number.")

