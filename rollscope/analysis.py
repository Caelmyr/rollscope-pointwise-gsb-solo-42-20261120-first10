from collections import OrderedDict
from functools import cached_property
from threading import RLock

from .acquisition import measure_acquisition
from .calibration import calibrate
from .coverage import inspect_coverage
from .edges import track_edges
from .grid import build_grid
from .illumination import measure_illumination
from .marks import inspect_marks
from .motion import infer_motion, motion_values
from .presentation import extrema_indices
from .reference_passes import report_reference_passes
from .splices import inspect_splices
from .spots import detect_spots
from .streaks import measure_streaks
from .texture import measure_texture


SECTION_TITLES = {
    "overview": "Station overview",
    "sensor": "Raw acquisition",
    "flat": "Flat-field response",
    "coverage": "Sampling coverage",
    "edges": "Strip geometry",
    "spots": "Surface spots",
    "streaks": "Longitudinal streaks",
    "marks": "Repeat pitch",
    "splices": "Transverse joins",
    "texture": "Surface texture",
    "illumination": "Reflectance trend",
}


class AnalysisContext:
    def __init__(self, capture, settings, transverse_calibration=None):
        self.capture = capture
        self.settings = settings
        self.transverse_calibration = transverse_calibration

    @cached_property
    def calibrated(self):
        return calibrate(self.capture, self.settings.calibration)

    @cached_property
    def grid(self):
        return build_grid(
            self.capture,
            self.calibrated,
            self.settings.grid,
            self.transverse_calibration,
        )

    @cached_property
    def coverage(self):
        return inspect_coverage(self.grid)

    @cached_property
    def acquisition(self):
        return measure_acquisition(self.capture, self.grid.plan.gap_limit_mm)

    @cached_property
    def edges(self):
        return track_edges(self.grid, self.settings.edges)

    @cached_property
    def spots(self):
        return detect_spots(self.grid, self.edges, self.settings.spots)

    @cached_property
    def streaks(self):
        return measure_streaks(self.grid, self.edges, self.settings.streaks)

    @cached_property
    def marks(self):
        return inspect_marks(self.grid, self.edges, self.settings.marks)

    @cached_property
    def splices(self):
        return inspect_splices(self.grid, self.edges, self.settings.splices)

    @cached_property
    def texture(self):
        return measure_texture(self.grid, self.edges, self.settings.texture)

    @cached_property
    def illumination(self):
        return measure_illumination(self.grid, self.edges)


class AnalysisCache:
    def __init__(self, limit=3):
        self.limit = limit
        self.entries = OrderedDict()
        self.lock = RLock()

    def context(self, ref, sha, capture, settings, transverse_calibration=None):
        calibration_id = (
            transverse_calibration.calibration_id if transverse_calibration else None
        )
        key = (ref, sha, settings, calibration_id)
        with self.lock:
            context = self.entries.pop(key, None)
            if context is None:
                context = AnalysisContext(capture, settings, transverse_calibration)
            self.entries[key] = context
            while len(self.entries) > self.limit:
                self.entries.popitem(last=False)
        return context


def result_table(key, title, records):
    return {"key": key, "title": title, "rows": records}


def plot(title, records, x, y, unit, secondary=()):
    metrics = (y, *secondary)
    return {
        "title": title,
        "x": x,
        "y": list(metrics),
        "unit": unit,
        "records": records,
    }


class AnalysisPresenter:
    def __init__(self, context, max_points=320):
        self.context = context
        self.max_points = max_points

    def output(self, section):
        methods = {
            "overview": self.overview,
            "sensor": self.sensor,
            "flat": self.flat,
            "coverage": self.coverage,
            "edges": self.edges,
            "spots": self.spots,
            "streaks": self.streaks,
            "marks": self.marks,
            "splices": self.splices,
            "texture": self.texture,
            "illumination": self.illumination,
        }
        summary, tables, charts = methods[section]()
        return {
            "section": section,
            "name": SECTION_TITLES[section],
            "summary": summary,
            "tables": tables,
            "charts": charts,
        }

    def overview(self):
        context = self.context
        grid = context.grid
        support = context.coverage.summary()
        result = grid.summary() | {
            "capture": context.capture.overview(),
            "encoder_gap_count": support["encoder_gap_count"],
            "longest_gap_mm": support["longest_gap_mm"],
        }
        table = result_table("gaps", "Unsampled station intervals", context.coverage.table())
        return result, [table], []

    def sensor(self):
        capture = self.context.capture
        report = self.context.acquisition
        timeline = report.series(capture.stations, self.max_points)
        transport = infer_motion(capture, report)
        speeds = motion_values(transport, self.max_points)
        charts = [
            plot("Encoder travel per line", timeline, "arrival_line", "step_mm", "mm"),
            plot("Clock delay per line", timeline, "arrival_line", "delay_ms", "ms"),
            plot("Transport speed", speeds, "elapsed_s", "speed_mm_s", "mm/s"),
        ]
        table = result_table("events", "Acquisition events", report.table())
        segments = result_table("transport", "Continuous transport spans", [item.summary() for item in transport])
        return report.summary, [table, segments], charts

    def flat(self):
        calibrated = self.context.calibrated
        columns = calibrated.reference.column_records(self.context.capture.transverse)
        faults = [column for column in columns if not column["valid"]]
        behavior, pass_stats, reference_drift = report_reference_passes(
            self.context.capture, calibrated.reference, self.context.settings.calibration
        )
        charts = [
            plot("Dark-to-white response", columns, "transverse_mm", "response_code", "ADU"),
            plot(
                "Reference variation",
                columns,
                "transverse_mm",
                "white_noise_code",
                "ADU",
                ("dark_noise_code",),
            ),
        ]
        charts.append(
            plot(
                "Reference drift across passes",
                reference_drift,
                "transverse_mm",
                "white_drift_code",
                "ADU",
                ("dark_drift_code",),
            )
        )
        tables = [
            result_table("flagged_columns", "Flagged detector columns", faults),
            result_table("dark_passes", "Dark reference passes", pass_stats["dark"]),
            result_table("white_passes", "White reference passes", pass_stats["white"]),
        ]
        return calibrated.summary() | behavior, tables, charts

    def coverage(self):
        report = self.context.coverage
        indices = extrema_indices([report.observed_pixels], self.max_points)
        gap_rows = report.table()
        series = report.series(indices)
        table = result_table("gaps", "Unsampled station intervals", gap_rows)
        chart = plot("Usable area by station", series, "station_mm", "usable_fraction", "fraction")
        return report.summary(), [table], [chart]

    def edges(self):
        report = self.context.edges
        indices = extrema_indices([report.width_mm, report.center_mm], self.max_points)
        series = report.row_table(indices)
        chart = plot("Strip width", series, "station_mm", "width_mm", "mm")
        position = plot(
            "Tracked edges",
            series,
            "station_mm",
            "left_mm",
            "mm",
            ("right_mm", "center_mm"),
        )
        table = result_table("width", "Width and center measurements", series)
        return report.summary(), [table], [chart, position]

    def spots(self):
        report = self.context.spots
        observations = report.table()
        table = result_table("spots", "Spot footprints", observations)
        return report.summary(), [table], []

    def streaks(self):
        report = self.context.streaks
        table = result_table("streaks", "Sustained streaks", report.table())
        return report.summary(), [table], []

    def marks(self):
        report = self.context.marks
        indices = extrema_indices([report.prominence, report.darkness], self.max_points)
        series = report.series(indices)
        chart = plot("Mark prominence", series, "station_mm", "prominence", "reflectance")
        marks = result_table("marks", "Detected marks", report.table())
        pitches = result_table("pitch", "Measured repeat intervals", report.intervals_table())
        return report.summary(), [marks, pitches], [chart]

    def splices(self):
        report = self.context.splices
        indices = extrema_indices([report.score, report.covered_fraction], self.max_points)
        series = report.series(indices)
        chart = plot("Transverse join score", series, "station_mm", "score", "reflectance")
        table = result_table("splices", "Join candidates", report.table())
        return report.summary(), [table], [chart]

    def texture(self):
        report = self.context.texture
        directions = report.orientations()
        wavelengths = report.wavelength_bands()
        charts = [
            plot("Stripe orientation", directions, "from_deg", "fraction", "fraction"),
            plot("Wavelength bands", wavelengths, "from_mm", "fraction", "fraction"),
        ]
        table = result_table("texture_tiles", "Texture tiles", report.table())
        return report.summary(), [table], charts

    def illumination(self):
        report = self.context.illumination
        rows = extrema_indices([report.longitudinal], self.max_points)
        along = report.row_series(rows)
        cross_web = report.column_series()
        charts = [
            plot("Along the roll", along, "station_mm", "reflectance", "reflectance"),
            plot("Across the roll", cross_web, "transverse_mm", "reflectance", "reflectance"),
            plot("Material histogram", report.histogram(), "from", "count", "samples"),
        ]
        table = result_table("trend", "Station blocks", list(report.trend))
        return report.summary(), [table], charts


def inspect_summary(context):
    grid = context.grid.summary()
    edges = context.edges.summary()
    flat = context.calibrated.summary()
    calibrated_width = edges["calibrated_width_mm"]["median"]
    return {
        "stations": grid["stations"],
        "pitch_mm": grid["pitch_mm"],
        "known_fraction": grid["valid_fraction"],
        "gap_count": context.coverage.summary()["encoder_gap_count"],
        "width_mm": calibrated_width if calibrated_width is not None else edges["width_mm"]["median"],
        "width_calibrated": calibrated_width is not None,
        "uncalibrated_columns": grid.get("uncalibrated_columns", 0),
        "suspect_columns": flat["reference"]["rejected_columns"],
    }
