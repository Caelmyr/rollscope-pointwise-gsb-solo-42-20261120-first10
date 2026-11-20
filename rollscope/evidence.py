import csv
import hashlib
import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from . import __version__
from .analysis import AnalysisPresenter
from .capture_codec import canonical_json
from .domain import UserInputError
from .grain import presentation as grain_report, source_profile
from .inspection_sheet import window_sheet
from .raster import Layer, RasterSpec, png_bytes, preview_image
from .window_review import WindowCells, review_window, row_provenance, source_rows


EVIDENCE_FORMAT = "rollscope-evidence/1"
MAX_EVIDENCE_BYTES = 80 * 1024 * 1024
REPORT_KEYS = {
    "overview": Layer.STATION,
    "sensor": Layer.SENSOR,
    "flat": Layer.FLAT,
    "coverage": Layer.COVERAGE,
    "edges": Layer.EDGES,
    "spots": Layer.SPOTS,
    "streaks": Layer.STREAKS,
    "marks": Layer.MARKS,
    "splices": Layer.SPLICES,
    "texture": Layer.TEXTURE,
    "illumination": Layer.ILLUMINATION,
}


def table_fields(rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def csv_cell(value):
    if isinstance(value, (list, dict)):
        return canonical_json(value).decode("utf-8")
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def csv_bytes(rows, fields=()):
    buffer = io.StringIO(newline="")
    if not fields:
        fields = table_fields(rows)
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: csv_cell(value) for key, value in row.items()})
    return buffer.getvalue().encode("utf-8")


def npz_bytes(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


class Evidence:
    def __init__(self, kind, engine=__version__):
        self.members = {}
        self.kind = kind
        self.engine = engine

    def add(self, name, content):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name == "manifest.json":
            raise ValueError(f"Invalid evidence member: {name}")
        if name in self.members:
            raise ValueError(f"Duplicate evidence member: {name}")
        self.members[name] = content

    def table(self, name, rows, fields=()):
        self.add(name, csv_bytes(rows, fields))

    def json(self, name, document):
        self.add(name, canonical_json(document))

    def raster(self, name, context, spec, grain=None):
        image, geometry = preview_image(context, spec, grain)
        self.add(name, png_bytes(image))
        return geometry.as_dict()

    def make(self, binding):
        total = sum(map(len, self.members.values()))
        if total > MAX_EVIDENCE_BYTES:
            raise UserInputError("The evidence exceeds 80 MiB. Export a smaller review window.")
        manifest = {
            "format": EVIDENCE_FORMAT,
            "kind": self.kind,
            "engine": self.engine,
            "binding": binding,
            "members": [
                {"path": path, "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
                for path, value in self.members.items()
            ],
        }
        buffer = io.BytesIO()
        with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
            archive.writestr("manifest.json", canonical_json(manifest))
            for path, contents in self.members.items():
                archive.writestr(path, contents)
        return buffer.getvalue()


def grid_payload(context, cells):
    grid = context.grid
    return npz_bytes(
        reflectance=cells.values,
        valid=cells.valid,
        material=cells.material,
        station_mm=cells.stations,
        transverse_mm=cells.transverse,
        support=cells.row_support,
        row_pitch_mm=np.float64(grid.row_pitch_mm),
        column_pitch_mm=np.float64(grid.column_pitch_mm),
    )


def source_payload(context, cells, lines):
    capture = context.capture
    selection = np.array([record["source_line"] for record in lines], dtype=np.int64)
    columns = cells.columns
    selected_columns = np.arange(columns.start, columns.stop, dtype=np.int64)
    return npz_bytes(
        capture_lines=selection,
        capture_columns=selected_columns,
        raw=capture.raw[selection, columns],
        ticks=capture.ticks[selection],
        seconds=capture.seconds[selection],
        dark=capture.dark[:, columns],
        white=capture.white[:, columns],
    )


def window_package(record, context, window):
    cells = WindowCells.take(context, window.region)
    report = review_window(context, window)
    evidence = Evidence("window")
    evidence.json("review.json", report)
    evidence.json("settings.json", context.settings.as_dict())
    raster_meta = evidence.raster(
        "inspection.png", context, RasterSpec(Layer.SPOTS, window=window.region)
    )
    evidence.add("station-cells.npz", grid_payload(context, cells))
    lines = source_rows(context, cells.rows)
    evidence.table("source-lines.csv", lines)
    evidence.table("supports.csv", list(row_provenance(context.grid, cells.rows)))
    evidence.add("source-samples.npz", source_payload(context, cells, lines))
    evidence.table("spots.csv", report["spots"])
    evidence.table("gaps.csv", report["gaps"])
    evidence.table("station-profile.csv", report["profiles"]["station_bands"])
    evidence.table("transverse-profile.csv", report["profiles"]["transverse_bands"])
    evidence.json("raster-mapping.json", raster_meta)
    evidence.add("review-sheet.png", png_bytes(window_sheet(record, context, window)))
    return evidence.make(
        {
            "capture_sha256": record.sha256,
            "project_capture_id": record.id,
            "revision": record.state.revision,
            "window_id": window.id,
            "geometry": record.capture.metadata.geometry.as_dict(),
        }
    )


def export_result_tables(evidence, result):
    for table in result["tables"]:
        evidence.table(f"tables/{table['key']}.csv", table["rows"])
    for position, chart in enumerate(result["charts"], 1):
        evidence.table(f"series/{position:02d}-{chart['x']}.csv", chart["records"])


def snapshot_package(record, snapshot, context):
    if record.sha256 != snapshot.capture_sha:
        raise UserInputError("The inspection snapshot no longer matches the stored capture.")
    evidence = Evidence("snapshot", snapshot.version)
    evidence.json("inspection.json", snapshot.as_dict())
    export_result_tables(evidence, snapshot.result)
    if snapshot.version == __version__:
        spec = RasterSpec(REPORT_KEYS[snapshot.section])
        evidence.json("raster-mapping.json", evidence.raster("inspection.png", context, spec))
    return evidence.make(
        {
            "capture_sha256": record.sha256,
            "project_capture_id": record.id,
            "saved_from_revision": snapshot.source_revision,
            "snapshot_id": snapshot.id,
        }
    )


def current_package(record, context, section):
    result = AnalysisPresenter(context, max_points=900).output(section)
    evidence = Evidence("analysis")
    evidence.json("inspection.json", result)
    evidence.json("settings.json", context.settings.as_dict())
    export_result_tables(evidence, result)
    spec = RasterSpec(REPORT_KEYS[section])
    evidence.json("raster-mapping.json", evidence.raster("inspection.png", context, spec))
    return evidence.make(
        {
            "capture_sha256": record.sha256,
            "project_capture_id": record.id,
            "revision": record.state.revision,
            "analysis": section,
        }
    )


def tile_source_payload(source):
    """Raw scan lines that produced a window, with their own geometry and references.

    Station rows are samples on the analysis grid; the original scan lines are
    looked up through the encoder grouping so the exported codes are the ones
    the detector recorded, not resampled rows.
    """
    from .window_review import source_line_set

    context = source.context
    capture = context.capture
    rows, columns = source.slices()
    lines = np.array(source_line_set(context.grid, rows), dtype=np.int64)
    selected = np.arange(columns.start, columns.stop, dtype=np.int64)
    return npz_bytes(
        capture_lines=lines,
        capture_columns=selected,
        raw=capture.raw[np.ix_(lines, selected)],
        ticks=capture.ticks[lines],
        seconds=capture.seconds[lines],
        station_mm=capture.stations[lines],
        transverse_mm=capture.transverse[columns],
        dark=capture.dark[:, columns],
        white=capture.white[:, columns],
        grid_reflectance=context.grid.values[rows, columns],
        grid_valid=context.grid.valid[rows, columns],
        material=context.edges.material[rows, columns],
    )


def grain_package(record, context, comparison, settings, sources, region):
    evidence = Evidence("grain")
    evidence.json("inspection.json", grain_report(comparison))
    evidence.json("settings.json", settings.as_dict())
    evidence.json("selection.json", comparison.selection.as_dict())
    evidence.table("target-tiles.csv", list(comparison.rows))
    evidence.table("reference-a-tiles.csv", list(comparison.reference_a_rows))
    evidence.table("reference-b-tiles.csv", list(comparison.reference_b_rows))
    evidence.table(
        "envelopes.csv",
        [comparison.envelope_a.as_dict(), comparison.envelope_b.as_dict()],
    )
    bands = []
    tiles = []
    lines = []
    for key in ("target", "reference_a", "reference_b"):
        window = comparison.window(key)
        evidence.json(f"window-{key}.json", window.summary())
        for tile in window.tiles:
            tiles.append({"window": key} | tile.evidence())
            if not tile.measured:
                continue
            for band in tile.band_rows(window.wave_edges):
                bands.append(
                    {
                        "window": key,
                        "tile": tile.id,
                        "station_mm": tile.station_mm,
                        "transverse_mm": tile.transverse_mm,
                        "from_mm": band["from_mm"],
                        "to_mm": band["to_mm"],
                        "fraction": band["fraction"],
                    }
                )
        source = next((item for item in sources if item.role == key), None)
        if source is not None:
            evidence.add(f"source-{key}.npz", tile_source_payload(source))
            for tile in window.measured_tiles:
                lines.extend(
                    row | {"window": key}
                    for row in source_profile(comparison.contexts[key], window, tile)
                )
    evidence.table("tiles.csv", tiles)
    evidence.table("wavelength-bands.csv", bands)
    evidence.table("source-lines.csv", lines)
    spec = RasterSpec(Layer.GRAIN, window=region)
    evidence.json(
        "raster-mapping.json",
        evidence.raster("classification.png", context, spec, comparison),
    )
    return evidence.make(
        {
            "capture_sha256": record.sha256,
            "project_capture_id": record.id,
            "revision": record.state.revision,
            "selection": comparison.selection.as_dict(),
            "geometry": record.capture.metadata.geometry.as_dict(),
        }
    )
