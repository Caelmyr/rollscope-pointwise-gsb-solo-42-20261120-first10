from collections import OrderedDict
from dataclasses import replace
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
from .records import InspectionSnapshot, ReviewWindow, identifier, utc_now
from .smear import run_table, state_table, summary as smear_summary
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

    def render_spec(self, record, layer, overlays, window_id, recorded=False):
        if window_id is None or layer in ACQUISITION_LAYERS:
            return RasterSpec(layer, show_overlays=overlays)
        window = record.state.get_window(window_id)
        return RasterSpec(
            layer, show_overlays=overlays, window=window.region, recorded=recorded
        )

    def smear_windows(self, ref, expected, window_id=None):
        record, context = self.configured(ref, expected)
        if window_id is not None:
            window = record.state.get_window(window_id)
            reports = [window]
        else:
            reports = list(record.state.windows)
        payload = []
        for window in reports:
            report = context.smear_window(window.region)
            payload.append(
                {
                    "window": window.as_dict(),
                    "summary": smear_summary(report, context),
                    "states": report.state_counts(),
                    "runs": run_table(report),
                    "rows": state_table(report),
                }
            )
        return {"windows": payload, "revision": record.state.revision}

    def raster(self, record_id, expected, layer, overlays, window_id, recorded=False):
        record, context = self.configured(record_id, expected)
        spec = self.render_spec(record, layer, overlays, window_id, recorded)
        picture, mapping = preview_image(context, spec)
        return png_bytes(picture), mapping.as_dict(), record.state.revision

    def view(self, record_id, expected, layer, window_id, recorded=False):
        record, context = self.configured(record_id, expected)
        spec = self.render_spec(record, layer, True, window_id, recorded)
        return view_extent(context, spec) | {"revision": record.state.revision}

    def probe(self, record_id, expected, layer, window_id, recorded, x, y):
        record, context = self.configured(record_id, expected)
        spec = self.render_spec(record, layer, True, window_id, recorded)
        result = point_measurement(context, spec, x, y)
        return result | {"revision": record.state.revision}

    def profile(self, record_id, expected, layer, window_id, recorded, y, points):
        record, context = self.configured(record_id, expected)
        spec = self.render_spec(record, layer, True, window_id, recorded)
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
