from collections import OrderedDict
from dataclasses import replace
from threading import RLock

from . import __version__
from .analysis import AnalysisCache, AnalysisContext, AnalysisPresenter, inspect_summary
from .archive import project_export, read_project
from .capture_codec import read_capture
from .compare import snapshot_comparison
from .demo import build_demo
from .domain import MissingRecord, RevisionConflict, UserInputError
from .evidence import current_package, grain_package, snapshot_package, window_package
from .grain import (
    GrainSelection,
    GrainSource,
    WindowReference,
    compare_grain,
    presentation as grain_presentation,
    source_profile,
    tile_detail,
)
from .measure_export import export_ruler
from .probe import cross_section, point_measurement, view_extent
from .raster import ACQUISITION_LAYERS, Layer, RasterSpec, png_bytes, preview_image
from .ruler import Ruler, evaluate_ruler
from .records import InspectionSnapshot, ReviewWindow, identifier, utc_now
from .window_review import review_window


class CaptureService:
    def __init__(self, repository):
        self.repository = repository
        self.analysis = AnalysisCache()
        self.capture_cache = OrderedDict()
        self.capture_lock = RLock()
        self.grain_cache = OrderedDict()
        self.grain_lock = RLock()
        self.grain_limit = 3

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
        if project.grain_selection is not None:
            selection = GrainSelection(
                *(
                    WindowReference(record.id, project.grain_selection.reference_for(role).window)
                    for role in ("reference_a", "reference_b", "target")
                )
            )
            self.repository.set_grain_selection(record.id, selection)
        with self.capture_lock:
            self.remember(record)
        return record

    def export_project(self, ref):
        record, snapshots = self.repository.project_records(ref)
        return project_export(record, snapshots, self.repository.grain_selection(ref))

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

    def render_spec(self, record, layer, overlays, window_id):
        if layer == Layer.GRAIN:
            selection = self.repository.grain_selection(record.id)
            target = selection.target if selection else None
            if target is not None and target.capture == record.id:
                window = record.state.get_window(target.window)
                return RasterSpec(layer, show_overlays=overlays, window=window.region)
            return RasterSpec(layer, show_overlays=overlays)
        if window_id is None or layer in ACQUISITION_LAYERS:
            return RasterSpec(layer, show_overlays=overlays)
        window = record.state.get_window(window_id)
        return RasterSpec(layer, show_overlays=overlays, window=window.region)

    def raster(self, record_id, expected, layer, overlays, window_id):
        record, context = self.configured(record_id, expected)
        spec = self.render_spec(record, layer, overlays, window_id)
        grain = None
        if layer == Layer.GRAIN:
            grain = self.grain_for(record)
            if grain is not None and grain.target.capture_ref != record.id:
                raise UserInputError(
                    "Open the target capture to draw its grain classification."
                )
        picture, mapping = preview_image(context, spec, grain)
        return png_bytes(picture), mapping.as_dict(), record.state.revision

    def view(self, record_id, expected, layer, window_id):
        record, context = self.configured(record_id, expected)
        spec = self.render_spec(record, layer, True, window_id)
        return view_extent(context, spec) | {"revision": record.state.revision}

    def probe(self, record_id, expected, layer, window_id, x, y):
        record, context = self.configured(record_id, expected)
        spec = self.render_spec(record, layer, True, window_id)
        result = point_measurement(context, spec, x, y, self.grain_for(record))
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

    def grain_for(self, record):
        """The stored comparison for one capture, or None when it cannot be built."""
        selection = self.repository.grain_selection(record.id)
        if selection is None or not selection.configured:
            return None
        sources = self.resolve_sources(selection)
        key = (
            record.id,
            selection.signature(),
            record.state.settings.grain,
            tuple(
                (source.capture_ref, source.window.id, source.context.settings)
                for source in sources
            ),
        )
        with self.grain_lock:
            cached = self.grain_cache.pop(key, None)
            if cached is None:
                cached = compare_grain(record.state.settings.grain, sources)
            self.grain_cache[key] = cached
            while len(self.grain_cache) > self.grain_limit:
                self.grain_cache.popitem(last=False)
        return cached

    def set_grain_selection(self, ref, expected, payload):
        record = self.read(ref, expected)
        selection = GrainSelection.parse(payload)
        if selection.configured:
            sources = self.resolve_sources(selection)
            compare_grain(record.state.settings.grain, sources)
        stored = self.repository.set_grain_selection(ref, selection)
        return {"selection": stored.as_dict(), "state": record.state.as_dict()}

    def resolve_sources(self, selection):
        references = []
        for role in ("reference_a", "reference_b", "target"):
            reference = selection.reference_for(role)
            if reference is None:
                continue
            record = self.read(reference.capture)
            window = record.state.get_window(reference.window)
            references.append(
                GrainSource(role, record.id, self.context_for(record), window)
            )
        return references

    def window_catalog(self):
        return [
            {
                "capture_ref": item["capture_ref"],
                "capture_name": item["capture_name"],
                "window_id": window.id,
                "label": window.label,
                "status": window.status.value,
                "note": window.note,
                "region": window.region.as_dict(),
            }
            for item in self.repository.list_window_catalog()
            for window in item["windows"]
        ]

    def grain_comparison(self, ref):
        record = self.read(ref)
        selection = self.repository.grain_selection(ref)
        catalog = self.window_catalog()
        if selection is None or not selection.configured:
            return {
                "configured": False,
                "selection": selection.as_dict() if selection else None,
                "catalog": catalog,
            }
        comparison = self.grain_for(record)
        if comparison is None:
            raise UserInputError("A referenced grain window is missing or no longer exists.")
        output = grain_presentation(comparison)
        return output | {
            "configured": True,
            "revision": record.state.revision,
            "selection": comparison.selection.as_dict(),
            "target_capture": comparison.target.capture_ref,
            "target_capture_name": comparison.target.capture_name,
            "catalog": catalog,
        }

    def grain_export(self, ref):
        record = self.read(ref)
        comparison = self.grain_for(record)
        if comparison is None:
            raise UserInputError("Configure both signed-off grain references before exporting.")
        selection = comparison.selection
        target = self.read(selection.target.capture)
        window = target.state.get_window(selection.target.window)
        return grain_package(
            record,
            self.context_for(target),
            comparison,
            record.state.settings,
            self.resolve_sources(selection),
            window.region,
        )

    def grain_tile(self, ref, role, tile_id):
        if role not in ("target", "reference_a", "reference_b"):
            raise UserInputError("Choose the target or one of the reference windows.")
        comparison = self.grain_for(self.read(ref))
        if comparison is None:
            raise UserInputError("Configure both signed-off references before inspecting a tile.")
        detail = tile_detail(comparison, role, tile_id)
        window = comparison.window(role)
        tile = window.tiles[int(tile_id) - 1]
        detail["source_lines"] = source_profile(comparison.contexts[role], window, tile)
        return detail
