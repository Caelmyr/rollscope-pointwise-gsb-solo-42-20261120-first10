import numpy as np

from .calibration import safe_value
from .domain import UserInputError
from .grid import Support
from .lineage import detailed_lineage
from .presentation import extrema_indices
from .raster import (
    ACQUISITION_LAYERS,
    WINDOW_LAYERS,
    Layer,
    position_mapping,
    smear_window,
)
from .smear import SmearState, STATE_MEANING


def point_measurement(context, spec, x, y):
    mapping = position_mapping(context, spec)
    row, column = mapping.cell(x, y)
    if spec.layer == Layer.SMEAR:
        result = smear_point(context, spec, row, column)
        result.update(
            layer=spec.layer.value,
            column=column,
            transverse_mm=float(context.capture.transverse[column]),
            coordinate=mapping.as_dict(),
        )
        return result
    result = {
        "layer": spec.layer.value,
        "column": column,
        "transverse_mm": float(context.capture.transverse[column]),
        "coordinate": mapping.as_dict(),
        "lineage": detailed_lineage(context, row, column, spec.layer.value),
    }
    if spec.layer in ACQUISITION_LAYERS:
        return result | acquisition_point(context, row, column)
    return result | station_point(context, row, column, spec.layer)


def smear_point(context, spec, row, column):
    if spec.window is None:
        raise UserInputError("Motion-smear compensation applies to a saved review window.")
    report = context.smear_window(spec.window)
    local = row - report.first_row
    first_column = int(
        np.searchsorted(context.grid.transverse, report.transverse[0], side="left")
    )
    cell = column - first_column
    state = SmearState(int(report.state[local]))
    return {
        "row": row,
        "station_mm": float(report.stations[local]),
        "coverage": state.name.lower(),
        "coverage_meaning": STATE_MEANING[state],
        "valid": bool(np.isfinite(report.restored[local, cell])),
        "reflectance": safe_value(report.restored[local, cell]),
        "recorded_reflectance": safe_value(report.original[local, cell]),
        "displayed": "recorded" if spec.recorded else "restored",
        "blur_rows": safe_value(report.blur_rows[local]),
        "exposure_distance_mm": safe_value(report.blur_mm[local]),
        "speed_mm_s": safe_value(report.speed_mm_s[local]),
        "exposure_ms": safe_value(report.exposure_ms[local]),
        "lineage": smear_lineage(context, report, local, cell),
    }


def smear_lineage(context, report, local, column):
    """Recorded source lines behind the restored cell, plus their exposure data."""
    row = report.first_row + local
    grid = context.grid
    support = Support(grid.plan.support[row])
    records = grid.provenance(row, column)
    sources = []
    profile = report.profile
    for group in records["sources"]:
        for line in range(group["first_line"], group["first_line"] + group["line_count"]):
            sources.append(
                {
                    "source_line": int(line),
                    "encoder_tick": int(context.capture.ticks[line]),
                    "capture_time_s": float(context.capture.seconds[line]),
                    "sensor_code": int(context.capture.raw[line, column]),
                    "reflectance": safe_value(context.calibrated.values[line, column]),
                    "group_weight": float(group["weight"]),
                    "speed_known": bool(profile.known[line]),
                    "speed_mm_s": safe_value(profile.speed_mm_s[line]),
                    "exposure_ms": safe_value(profile.exposure_ms[line]),
                    "exposure_distance_mm": safe_value(
                        profile.exposure_distance_mm()[line]
                    ),
                }
            )
    return {
        "kind": "motion_smear",
        "row": int(row),
        "station_mm": float(grid.stations[row]),
        "grid_coverage": support.name.lower(),
        "recorded_reflectance": safe_value(report.original[local, column]),
        "restored_reflectance": safe_value(report.restored[local, column]),
        "state": SmearState(int(report.state[local])).name.lower(),
        "line_count": len(sources),
        "segments": [{"kind": "recorded_source_lines", "rows": sources}],
    }


def acquisition_point(context, row, column):
    capture = context.capture
    corrected = context.calibrated
    profile = corrected.reference
    return {
        "row": row,
        "source_line": row,
        "encoder_tick": int(capture.ticks[row]),
        "time_s": float(capture.seconds[row]),
        "station_mm": float(capture.stations[row]),
        "sensor_code": int(capture.raw[row, column]),
        "dark_code": float(profile.dark[column]),
        "white_code": float(profile.white[column]),
        "reference_faults": list(profile.fault_names(column)),
        "sample_saturated": bool(corrected.clipped[row, column]),
        "valid": bool(corrected.valid[row, column]),
        "reflectance": safe_value(corrected.values[row, column]),
    }


def station_point(context, row, column, layer):
    grid = context.grid
    result = grid.provenance(row, column) | {
        "valid": bool(grid.valid[row, column]),
        "reflectance": safe_value(grid.values[row, column]),
    }
    if layer == Layer.SPOTS:
        result.update(context.spots.pixel(row, column))
    if layer == Layer.STREAKS:
        report = context.streaks
        result.update(
            detectable=bool(report.evidence[row, column]),
            signed_contrast=safe_value(report.residual[row, column]),
        )
    return result


def selected_axis(context, spec):
    capture = context.capture
    if spec.layer in ACQUISITION_LAYERS:
        return capture.stations
    return context.grid.stations


def cross_section(context, spec, y, points):
    mapping = position_mapping(context, spec)
    row, _ = mapping.cell(0, y)
    cols = np.arange(mapping.columns.start, mapping.columns.stop)
    position = selected_axis(context, spec)[row]
    if spec.layer in ACQUISITION_LAYERS:
        values, validity = acquisition_profile(context, row, cols, spec.layer)
        coverage = "acquisition"
    elif spec.layer == Layer.SMEAR:
        report = smear_window(context, spec)
        values = report.restored[row - report.first_row, cols - mapping.columns.start]
        validity = np.isfinite(values)
        coverage = SmearState(int(report.state[row - report.first_row])).name.lower()
    else:
        values, validity = station_profile(context, row, cols, spec.layer)
        coverage = Support(context.grid.plan.support[row]).name.lower()

    indices = extrema_indices([np.where(validity, values, np.nan)], points)
    measurements = [
        {
            "column": int(cols[index]),
            "transverse_mm": float(context.capture.transverse[cols[index]]),
            "value": safe_value(values[index]),
            "valid": bool(validity[index]),
        }
        for index in indices
    ]
    return {
        "station_mm": float(position),
        "row": row,
        "coverage": coverage,
        "value_unit": measure_unit(spec.layer),
        "measurements": measurements,
    }


def acquisition_profile(context, row, cols, layer):
    if layer == Layer.SENSOR:
        values = context.capture.raw[row, cols]
        return values, np.ones(values.shape, dtype=bool)
    corrected = context.calibrated
    return corrected.values[row, cols], corrected.valid[row, cols]


def station_profile(context, row, cols, layer):
    if layer == Layer.SPOTS:
        report = context.spots
        return report.residual[row, cols], report.detectable[row, cols]
    if layer == Layer.STREAKS:
        report = context.streaks
        values = report.residual[row, cols]
        return values, np.isfinite(values)
    grid = context.grid
    return grid.values[row, cols], grid.valid[row, cols]


def measure_unit(layer):
    if layer == Layer.SENSOR:
        return "ADU"
    if layer in (Layer.SPOTS, Layer.STREAKS):
        return "contrast"
    return "reflectance"


def view_extent(context, spec):
    mapping = position_mapping(context, spec)
    if spec.layer in ACQUISITION_LAYERS:
        y0 = mapping.rows.start
        y1 = mapping.rows.stop - 1
        y_units = "line"
    else:
        stations = context.grid.stations
        y0 = float(stations[mapping.rows.start])
        y1 = float(stations[mapping.rows.stop - 1])
        y_units = "mm"
    transverse = context.capture.transverse
    return mapping.as_dict() | {
        "y_from": y0,
        "y_to": y1,
        "y_unit": y_units,
        "x_from_mm": float(transverse[mapping.columns.start]),
        "x_to_mm": float(transverse[mapping.columns.stop - 1]),
    }

