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
    read_enum,
    reject_unknown,
)
from .settings import AnalysisSettings
from .trim import TrimObjective


MAX_WINDOWS = 40
MAX_SNAPSHOTS = 40
MAX_TRIM_PLANS = 8
MAX_TRIM_SPOTS = 3


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def identifier():
    return uuid4().hex[:16]


@dataclass(frozen=True)
class TrimPlanRecord:
    id: str
    label: str
    objective: str
    margin_mm: float
    minimum_length_mm: float
    retain_start_mm: float | None
    retain_end_mm: float | None
    spot_ids: tuple
    plan: dict
    spot_settings: dict
    capture_sha: str
    source_revision: int
    created_at: str

    @classmethod
    def create(cls, data, plan, settings, capture_sha, revision):
        record = expect_object(data, "Trim confirmation")
        exact_keys(
            record,
            ("label", "objective", "margin_mm", "minimum_length_mm",
             "retain_start_mm", "retain_end_mm", "spot_ids"),
            "Trim confirmation",
        )
        spot_ids = record["spot_ids"]
        if not isinstance(spot_ids, list) or not all(type(item) is int for item in spot_ids):
            raise UserInputError("Spot selections must be whole-number spot identifiers.")
        if not 1 <= len(spot_ids) <= MAX_TRIM_SPOTS:
            raise UserInputError(
                f"A confirmed trim must remove between 1 and {MAX_TRIM_SPOTS} spots."
            )
        if len(set(spot_ids)) != len(spot_ids):
            raise UserInputError("Each spot can only be selected once.")
        objective = read_enum(TrimObjective, record["objective"], "Trim objective").value
        return cls(
            identifier(),
            clean_text(record["label"], "Trim plan name", 75),
            objective,
            float(record["margin_mm"]),
            float(record["minimum_length_mm"]),
            record["retain_start_mm"],
            record["retain_end_mm"],
            tuple(spot_ids),
            plan,
            asdict(settings.spots),
            capture_sha,
            revision,
            utc_now(),
        )

    @classmethod
    def stored(cls, data):
        return cls(
            data["id"],
            data["label"],
            data["objective"],
            float(data["margin_mm"]),
            float(data["minimum_length_mm"]),
            data["retain_start_mm"],
            data["retain_end_mm"],
            tuple(data["spot_ids"]),
            data["plan"],
            data["spot_settings"],
            data["capture_sha"],
            int(data["source_revision"]),
            data["created_at"],
        )

    def request_payload(self):
        return {
            "margin_mm": self.margin_mm,
            "minimum_length_mm": self.minimum_length_mm,
            "retain_start_mm": self.retain_start_mm,
            "retain_end_mm": self.retain_end_mm,
            "spot_ids": list(self.spot_ids),
        }

    def as_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "objective": self.objective,
            "margin_mm": self.margin_mm,
            "minimum_length_mm": self.minimum_length_mm,
            "retain_start_mm": self.retain_start_mm,
            "retain_end_mm": self.retain_end_mm,
            "spot_ids": list(self.spot_ids),
            "plan": self.plan,
            "spot_settings": self.spot_settings,
            "capture_sha": self.capture_sha,
            "source_revision": self.source_revision,
            "created_at": self.created_at,
        }


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
    trim_plans: tuple
    notes: str
    updated_at: str

    @classmethod
    def create(cls, timestamp):
        return cls(0, AnalysisSettings.defaults(), (), (), "", timestamp)

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["revision"],
            AnalysisSettings.parse(data["settings"]),
            tuple(ReviewWindow.stored(item) for item in data["windows"]),
            tuple(TrimPlanRecord.stored(item) for item in data.get("trim_plans", ())),
            data["notes"],
            data["updated_at"],
        )

    def as_dict(self):
        return {
            "revision": self.revision,
            "settings": self.settings.as_dict(),
            "windows": [window.as_dict() for window in self.windows],
            "trim_plans": [plan.as_dict() for plan in self.trim_plans],
            "notes": self.notes,
            "updated_at": self.updated_at,
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

    def get_trim_plan(self, ref):
        for plan in self.trim_plans:
            if plan.id == ref:
                return plan
        raise MissingRecord("That confirmed trim plan no longer exists.")

    def add_trim_plan(self, plan):
        if len(self.trim_plans) >= MAX_TRIM_PLANS:
            raise UserInputError(
                f"Saving another trim plan would exceed the limit of {MAX_TRIM_PLANS}."
            )
        return replace(self, trim_plans=self.trim_plans + (plan,))

    def remove_trim_plan(self, ref):
        self.get_trim_plan(ref)
        return replace(
            self, trim_plans=tuple(item for item in self.trim_plans if item.id != ref)
        )

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
