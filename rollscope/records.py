import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from uuid import uuid4

from .domain import (
    Classification,
    MissingRecord,
    StationWindow,
    UserInputError,
    clean_text,
    exact_keys,
    expect_object,
    finite_number,
    read_enum,
    reject_unknown,
)
from .settings import AnalysisSettings


MAX_WINDOWS = 40
MAX_SNAPSHOTS = 40
MAX_LOG_EVENTS = 400
MAX_LOG_ANCHORS = 60
MAX_LOG_TEXT = 30000
EVENT_ID = re.compile(r"e[0-9]{1,3}")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def identifier():
    return uuid4().hex[:16]


@dataclass(frozen=True)
class ReviewWindow:
    id: str
    label: str
    note: str
    status: Classification
    region: StationWindow

    @classmethod
    def from_request(cls, data, capture, ref=None):
        data = expect_object(data, "Review window")
        exact_keys(data, ("label", "note", "status", "region"), "Review window")
        return cls(
            ref or identifier(),
            clean_text(data["label"], "Window name", 75),
            clean_text(data["note"], "Window note", 1000, True),
            read_enum(Classification, data["status"], "Window status"),
            StationWindow.parse(data["region"]).within(capture),
        )

    @classmethod
    def stored(cls, data):
        return cls(
            data["id"],
            data["label"],
            data["note"],
            Classification(data["status"]),
            StationWindow.parse(data["region"]),
        )

    def as_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "note": self.note,
            "status": self.status.value,
            "region": self.region.as_dict(),
        }


@dataclass(frozen=True)
class LogEvent:
    id: str
    kind: str
    note: str
    start_text: str
    end_text: str
    start_s: float
    end_s: float

    @classmethod
    def stored(cls, data):
        data = expect_object(data, "Stored log record")
        exact_keys(
            data,
            ("id", "kind", "note", "start_text", "end_text", "start_s", "end_s"),
            "Stored log record",
        )
        ref = clean_text(data["id"], "Log record id", 12)
        if EVENT_ID.fullmatch(ref) is None:
            raise UserInputError("A stored log record id is invalid.")
        event = cls(
            ref,
            clean_text(data["kind"], "Log record kind", 40),
            clean_text(data["note"], "Log record note", 200, True),
            clean_text(data["start_text"], "Log record start", 60),
            clean_text(data["end_text"], "Log record end", 60),
            finite_number(data["start_s"], "Log record start seconds", -1e12, 1e12),
            finite_number(data["end_s"], "Log record end seconds", -1e12, 1e12),
        )
        if event.end_s < event.start_s:
            raise UserInputError("A stored log record ends before it starts.")
        return event

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EventAnchor:
    event_id: str
    station_start_mm: float
    station_end_mm: float

    @classmethod
    def make(cls, data, event_ids, capture=None):
        data = expect_object(data, "Event anchor")
        exact_keys(
            data, ("event_id", "station_start_mm", "station_end_mm"), "Event anchor"
        )
        event_id = clean_text(data["event_id"], "Anchor record", 12)
        if event_id not in event_ids:
            raise UserInputError(f"An anchor refers to unknown log record {event_id}.")
        anchor = cls(
            event_id,
            finite_number(data["station_start_mm"], "Anchor start", -1e8, 1e8),
            finite_number(data["station_end_mm"], "Anchor end", -1e8, 1e8),
        )
        if anchor.station_start_mm >= anchor.station_end_mm:
            raise UserInputError("An anchor station range must have positive length.")
        if capture is not None:
            overview = capture.overview()
            if (
                anchor.station_start_mm < overview["station_start_mm"]
                or anchor.station_end_mm > overview["station_end_mm"]
            ):
                raise UserInputError(
                    "Anchor station ranges must lie inside the captured station range."
                )
        return anchor

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LogAlignment:
    log_text: str
    events: tuple
    anchors: tuple
    certainty_mm: float

    @classmethod
    def build(cls, data, events, capture):
        data = expect_object(data, "Log alignment")
        exact_keys(data, ("log_text", "anchors", "certainty_mm"), "Log alignment")
        log_text = clean_text(data["log_text"], "Operation log", MAX_LOG_TEXT)
        certainty = finite_number(data["certainty_mm"], "Certainty band", 0.1, 5000)
        return cls(log_text, events, read_anchors(data["anchors"], events, capture), certainty)

    @classmethod
    def stored(cls, data, capture=None):
        data = expect_object(data, "Stored log alignment")
        exact_keys(
            data, ("log_text", "events", "anchors", "certainty_mm"), "Stored log alignment"
        )
        raw_events = data["events"]
        if not isinstance(raw_events, list) or not 1 <= len(raw_events) <= MAX_LOG_EVENTS:
            raise UserInputError("The stored log alignment has an invalid record list.")
        events = tuple(LogEvent.stored(item) for item in raw_events)
        if len({event.id for event in events}) != len(events):
            raise UserInputError("The stored log alignment repeats a record id.")
        return cls(
            clean_text(data["log_text"], "Operation log", MAX_LOG_TEXT),
            events,
            read_anchors(data["anchors"], events, capture),
            finite_number(data["certainty_mm"], "Certainty band", 0.1, 5000),
        )

    def as_dict(self):
        return {
            "log_text": self.log_text,
            "certainty_mm": self.certainty_mm,
            "events": [event.as_dict() for event in self.events],
            "anchors": [anchor.as_dict() for anchor in self.anchors],
        }


def read_anchors(raw, events, capture):
    if not isinstance(raw, list) or len(raw) > MAX_LOG_ANCHORS:
        raise UserInputError(f"Provide at most {MAX_LOG_ANCHORS} anchors.")
    event_ids = {event.id for event in events}
    anchors = tuple(EventAnchor.make(item, event_ids, capture) for item in raw)
    if len({anchor.event_id for anchor in anchors}) != len(anchors):
        raise UserInputError("Each log record can carry at most one anchor.")
    return anchors


@dataclass(frozen=True)
class CaptureState:
    revision: int
    settings: AnalysisSettings
    windows: tuple
    notes: str
    updated_at: str
    alignment: LogAlignment | None = None

    @classmethod
    def create(cls, timestamp):
        return cls(0, AnalysisSettings.defaults(), (), "", timestamp)

    @classmethod
    def from_dict(cls, data):
        alignment = data.get("alignment")
        return cls(
            data["revision"],
            AnalysisSettings.parse(data["settings"]),
            tuple(ReviewWindow.stored(item) for item in data["windows"]),
            data["notes"],
            data["updated_at"],
            LogAlignment.stored(alignment) if alignment is not None else None,
        )

    def as_dict(self):
        return {
            "revision": self.revision,
            "settings": self.settings.as_dict(),
            "windows": [window.as_dict() for window in self.windows],
            "notes": self.notes,
            "updated_at": self.updated_at,
            "alignment": self.alignment.as_dict() if self.alignment else None,
        }

    def settings_patch(self, data):
        data = expect_object(data, "Capture settings")
        reject_unknown(data, ("section", "values"), "Capture settings")
        if not isinstance(data.get("section"), str):
            raise UserInputError("Provide one settings section.")
        settings = self.settings.with_section(data["section"], data.get("values"))
        return replace(self, settings=settings)

    def with_notes(self, value):
        notes = clean_text(value, "Capture notes", 4000, True)
        return replace(self, notes=notes)

    def with_alignment(self, alignment):
        return replace(self, alignment=alignment)

    def get_window(self, ref):
        for window in self.windows:
            if window.id == ref:
                return window
        raise MissingRecord("The review window no longer exists.")

    def add_window(self, window):
        if len(self.windows) >= MAX_WINDOWS:
            raise UserInputError(f"Opening another window would exceed the limit of {MAX_WINDOWS}.")
        return replace(self, windows=self.windows + (window,))

    def change_window(self, new):
        self.get_window(new.id)
        return replace(
            self, windows=tuple(new if item.id == new.id else item for item in self.windows)
        )

    def remove_window(self, ref):
        self.get_window(ref)
        return replace(self, windows=tuple(item for item in self.windows if item.id != ref))

    def next_revision(self, timestamp):
        return replace(self, revision=self.revision + 1, updated_at=timestamp)


@dataclass(frozen=True)
class InspectionSnapshot:
    id: str
    label: str
    section: str
    settings: AnalysisSettings
    result: dict
    capture_sha: str
    source_revision: int
    version: str
    timestamp: str

    @classmethod
    def stored(cls, data):
        return cls(
            data["id"],
            data["label"],
            data["section"],
            AnalysisSettings.parse(data["settings"]),
            data["result"],
            data["capture_sha"],
            data["source_revision"],
            data["version"],
            data["timestamp"],
        )

    def summary(self):
        return {
            "id": self.id,
            "label": self.label,
            "section": self.section,
            "capture_sha": self.capture_sha,
            "source_revision": self.source_revision,
            "version": self.version,
            "timestamp": self.timestamp,
        }

    def as_dict(self):
        return self.summary() | {
            "settings": self.settings.as_dict(),
            "result": self.result,
        }
