from dataclasses import dataclass, replace
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
    read_enum,
    reject_unknown,
)
from .settings import AnalysisSettings
from .transverse import TransverseCalibration


MAX_WINDOWS = 40
MAX_SNAPSHOTS = 40


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
class CaptureState:
    revision: int
    settings: AnalysisSettings
    windows: tuple
    notes: str
    updated_at: str
    transverse_calibration: TransverseCalibration | None = None

    @classmethod
    def create(cls, timestamp):
        return cls(0, AnalysisSettings.defaults(), (), "", timestamp, None)

    @classmethod
    def from_dict(cls, data):
        calibration = data.get("transverse_calibration")
        return cls(
            data["revision"],
            AnalysisSettings.parse(data["settings"]),
            tuple(ReviewWindow.stored(item) for item in data["windows"]),
            data["notes"],
            data["updated_at"],
            TransverseCalibration.parse(calibration) if calibration else None,
        )

    def as_dict(self):
        return {
            "revision": self.revision,
            "settings": self.settings.as_dict(),
            "windows": [window.as_dict() for window in self.windows],
            "notes": self.notes,
            "updated_at": self.updated_at,
            "transverse_calibration": (
                self.transverse_calibration.as_dict() if self.transverse_calibration else None
            ),
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

    def with_transverse_calibration(self, calibration):
        return replace(self, transverse_calibration=calibration)

    def without_transverse_calibration(self):
        return replace(self, transverse_calibration=None)

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
