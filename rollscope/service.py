from collections import OrderedDict
from dataclasses import asdict, replace
from threading import RLock

from . import __version__
from .analysis import AnalysisCache, AnalysisContext, AnalysisPresenter, inspect_summary
from .archive import project_export, read_project
from .capture_codec import read_capture
from .compare import snapshot_comparison
from .demo import build_demo
from .domain import RevisionConflict, UserInputError
from .evidence import current_package, snapshot_package, window_package
from .measure_export import export_ruler
from .probe import cross_section, point_measurement, view_extent
from .raster import ACQUISITION_LAYERS, RasterSpec, png_bytes, preview_image
from .ruler import Ruler, evaluate_ruler
from .records import InspectionSnapshot, ReviewWindow, TrimPlanRecord, identifier, utc_now
from .trim import TrimObjective, TrimPlanner, TrimRequest, evaluate_trim
from .trim_export import export_trim_plan, trim_view
from .window_review import review_window


class CaptureService:
    def __init__(self, repository):
        self.repository = repository
        self.analysis = AnalysisCache()
        self.capture_cache = OrderedDict()
        self.capture_lock = RLock()

    def seed_demo(self):
        if not self.repository.list_captures():
            self.make_demo("cold-rolled")

    def remember(self, record):
        self.capture_cache[record.id] = record
        while len(self.capture_cache) > 4:
            self.capture_cache.popitem(last=False)

    def make_demo(self, name):
        if not isinstance(name, str):
            raise UserInputError("Choose a synthetic capture.")
        record = self.repository.add_capture(build_demo(name))
        with self.capture_lock:
            self.remember(record)
        return record

    def import_bundle(self, data, original_name):
        capture = read_capture(data)
        record = self.repository.add_capture(capture, original_name, data)
        with self.capture_lock:
            self.remember(record)
        return record

    def import_project(self, payload):
        project = read_project(payload)
        AnalysisContext(project.capture, project.state.settings).grid
        record = self.repository.restore(project)
        with self.capture_lock:
            self.remember(record)
        return record

    def export_project(self, ref):
        record, snapshots = self.repository.project_records(ref)
        return project_export(record, snapshots)

    def read(self, ref, expected=None):
        with self.capture_lock:
            record = self.capture_cache.pop(ref, None)
            if record is None:
                updated = self.repository.get_capture(ref)
            else:
                fresh = self.repository.get_state(ref)
                updated = replace(record, state=fresh)
            self.remember(updated)
        if expected is not None and updated.state.revision != expected:
            raise RevisionConflict(updated.state.revision)
        return updated

    def context_for(self, record, settings=None):
        return self.analysis.context(
            record.id,
            record.sha256,
            record.capture,
            settings or record.state.settings,
        )

    def capture_details(self, ref):
        record = self.read(ref)
        context = self.context_for(record)
        status = inspect_summary(context)
        return record.manifest() | {
            "state": record.state.as_dict(),
            "quick_readings": status,
            "snapshots": self.repository.list_snapshots(ref),
            "trim_plans": [plan.as_dict() for plan in record.state.trim_plans],
        }

    def configured(self, ref, expected):
        record = self.read(ref, expected)
        return record, self.context_for(record)

    def inspect(self, ref, expected, section, points):
        record, context = self.configured(ref, expected)
        output = AnalysisPresenter(context, points).output(section)
        return output | {"revision": record.state.revision}

    def patch_settings(self, ref, expected, payload):
        record = self.read(ref, expected)
        candidate = record.state.settings_patch(payload)
        if candidate.settings.grid != record.state.settings.grid:
            self.context_for(record, candidate.settings).grid
        updated = self.repository.mutate(ref, expected, lambda current: candidate)
        status = inspect_summary(self.context_for(record, updated.settings))
        return {"state": updated.as_dict(), "quick_readings": status}

    def set_notes(self, ref, expected, notes):
        state = self.repository.mutate(ref, expected, lambda current: current.with_notes(notes))
        return state.as_dict()

    def write_window(self, ref, expected, data, window_id=None):
        record, context = self.configured(ref, expected)
        window = ReviewWindow.from_request(data, record.capture, window_id)
        window.region.indices(context.grid.stations, context.grid.transverse)
        if window_id is None:
            transform = lambda current: current.add_window(window)
        else:
            transform = lambda current: current.change_window(window)
        new_state = self.repository.mutate(ref, expected, transform)
        return {"state": new_state.as_dict(), "window": window.as_dict()}

    def delete_window(self, ref, expected, window_id):
        result = self.repository.mutate(ref, expected, lambda state: state.remove_window(window_id))
        return result.as_dict()

    def window_result(self, ref, expected, window_id):
        record, context = self.configured(ref, expected)
        window = record.state.get_window(window_id)
        result = review_window(context, window)
        return result | {"revision": record.state.revision}

    def window_export(self, ref, expected, window_id):
        record, context = self.configured(ref, expected)
        window = record.state.get_window(window_id)
        return window_package(record, context, window)

    def render_spec(self, record, layer, overlays, window_id, trim_plan=None):
        if window_id is None or layer in ACQUISITION_LAYERS:
            return RasterSpec(layer, show_overlays=overlays, trim_plan=trim_plan)
        window = record.state.get_window(window_id)
        return RasterSpec(
            layer, show_overlays=overlays, window=window.region, trim_plan=trim_plan
        )

    def raster(self, record_id, expected, layer, overlays, window_id, trim_plan=None):
        record, context = self.configured(record_id, expected)
        spec = self.render_spec(record, layer, overlays, window_id, trim_plan)
        picture, mapping = preview_image(context, spec)
        return png_bytes(picture), mapping.as_dict(), record.state.revision

    def view(self, record_id, expected, layer, window_id):
        record, context = self.configured(record_id, expected)
        spec = self.render_spec(record, layer, True, window_id)
        return view_extent(context, spec) | {"revision": record.state.revision}

    def probe(self, record_id, expected, layer, window_id, x, y):
        record, context = self.configured(record_id, expected)
        spec = self.render_spec(record, layer, True, window_id)
        result = point_measurement(context, spec, x, y)
        return result | {"revision": record.state.revision}

    def profile(self, record_id, expected, layer, window_id, y, points):
        record, context = self.configured(record_id, expected)
        spec = self.render_spec(record, layer, True, window_id)
        result = cross_section(context, spec, y, points)
        return result | {"revision": record.state.revision}

    def measure(self, ref, expected, endpoints, points):
        record, context = self.configured(ref, expected)
        ruler = Ruler.parse(endpoints, context.grid)
        payload = evaluate_ruler(context, ruler).as_dict(points)
        return payload | {"revision": record.state.revision}

    def export_measurement(self, ref, expected, endpoints):
        record, context = self.configured(ref, expected)
        ruler = Ruler.parse(endpoints, context.grid)
        return export_ruler(record, context, ruler)

    def save_snapshot(self, ref, expected, section, label):
        record, context = self.configured(ref, expected)
        result = AnalysisPresenter(context, 900).output(section)
        snapshot = InspectionSnapshot(
            identifier(),
            label,
            section,
            context.settings,
            result,
            record.sha256,
            record.state.revision,
            __version__,
            utc_now(),
        )
        state = self.repository.save_snapshot(ref, expected, snapshot)
        return {
            "snapshot": snapshot.summary(),
            "state": state.as_dict(),
        }

    def snapshot(self, ref, snapshot_id):
        return self.repository.get_snapshot(ref, snapshot_id).as_dict()

    def compare_snapshot(self, ref, expected, snapshot_id):
        record, context = self.configured(ref, expected)
        saved = self.repository.get_snapshot(ref, snapshot_id)
        current = AnalysisPresenter(context, 900).output(saved.section)
        return snapshot_comparison(saved, current, record.state.settings)

    def export_snapshot(self, ref, snapshot_id):
        record = self.read(ref)
        saved = self.repository.get_snapshot(ref, snapshot_id)
        return snapshot_package(record, saved, self.context_for(record, saved.settings))

    def export_analysis(self, ref, expected, section):
        record, context = self.configured(ref, expected)
        return current_package(record, context, section)

    def trim_options(self, ref, expected, payload):
        record, context = self.configured(ref, expected)
        result = evaluate_trim(context, payload)
        return result | {"revision": record.state.revision}

    def trim_preview(self, ref, expected, payload):
        record, context = self.configured(ref, expected)
        objective = read_trim_objective(payload)
        request = parse_trim_request(payload)
        planner = TrimPlanner(context)
        plan = planner.plan(objective, request)
        picture, mapping = trim_view(context, plan.as_dict())
        return (
            png_bytes(picture),
            mapping.as_dict(),
            record.state.revision,
            plan,
        )

    def confirm_trim(self, ref, expected, payload):
        record, context = self.configured(ref, expected)
        objective = read_trim_objective(payload)
        request = parse_trim_request(payload)
        if not request.spot_ids:
            raise UserInputError("Select at least one spot before confirming a trim plan.")
        planner = TrimPlanner(context)
        plan = planner.plan(objective, request)
        if not plan.feasible:
            messages = "; ".join(item["message"] for item in plan.conflicts)
            raise UserInputError("This trim plan cannot be confirmed: " + messages)
        saved = TrimPlanRecord.create(
            payload, plan.as_dict(), context.settings, record.sha256, expected
        )
        state = self.repository.mutate(ref, expected, lambda current: current.add_trim_plan(saved))
        return {
            "state": state.as_dict(),
            "trim_plan": saved.as_dict(),
        }

    def trim_plan(self, ref, trim_id):
        record = self.read(ref)
        saved = record.state.get_trim_plan(trim_id)
        context = self.context_for(record)
        current = self.recompute_saved_trim(context, saved)
        settings_match = asdict(context.settings.spots) == saved.spot_settings
        return {
            "saved": saved.as_dict(),
            "current": current,
            "settings_match": settings_match,
            "revision": record.state.revision,
        }

    def trim_plan_image(self, ref, trim_id, use_saved):
        record = self.read(ref)
        saved = record.state.get_trim_plan(trim_id)
        context = self.context_for(record)
        if use_saved:
            plan = saved.plan
        else:
            plan = self.recompute_saved_trim(context, saved)
        picture, mapping = trim_view(context, plan)
        return png_bytes(picture), mapping.as_dict(), record.state.revision

    def recompute_saved_trim(self, context, saved):
        try:
            return TrimPlanner(context).plan(
                TrimObjective(saved.objective),
                TrimRequest(
                    saved.margin_mm,
                    saved.minimum_length_mm,
                    saved.retain_start_mm,
                    saved.retain_end_mm,
                    saved.spot_ids,
                ),
            ).as_dict()
        except UserInputError as error:
            return {
                "objective": saved.objective,
                "feasible": False,
                "conflicts": [{"code": "recompute_failed", "message": str(error)}],
                "warnings": [],
                "selected_spots": [],
                "cut_regions": [],
                "segments": [],
                "retain": (
                    None
                    if saved.retain_start_mm is None
                    else {
                        "start_mm": saved.retain_start_mm,
                        "end_mm": saved.retain_end_mm,
                    }
                ),
                "cut_count": 0,
                "removed_length_mm": 0.0,
                "retained_length_mm": 0.0,
                "longest_segment_mm": 0.0,
                "shippable_length_mm": 0.0,
                "shippable_segment_count": 0,
                "total_length_mm": 0.0,
            }

    def delete_trim_plan(self, ref, expected, trim_id):
        state = self.repository.mutate(
            ref, expected, lambda current: current.remove_trim_plan(trim_id)
        )
        return state.as_dict()

    def export_trim(self, ref, trim_id):
        record = self.read(ref)
        saved = record.state.get_trim_plan(trim_id)
        context = self.context_for(record)
        return export_trim_plan(record, context, saved)


def read_trim_objective(payload):
    from .domain import read_enum

    data = payload if isinstance(payload, dict) else {}
    return read_enum(TrimObjective, data.get("objective", TrimObjective.LONGEST.value), "Trim objective")


def parse_trim_request(payload):
    return TrimRequest.parse(
        {
            key: payload[key]
            for key in (
                "margin_mm",
                "minimum_length_mm",
                "retain_start_mm",
                "retain_end_mm",
                "spot_ids",
            )
        }
    )
