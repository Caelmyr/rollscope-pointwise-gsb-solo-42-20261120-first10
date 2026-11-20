import numpy as np

from .calibration import safe_value
from .grid import Support
from .lineage import detailed_lineage
from .presentation import extrema_indices
from .raster import ACQUISITION_LAYERS, Layer, position_mapping


def point_measurement(context, spec, x, y):
    mapping = position_mapping(context, spec)
    row, column = mapping.cell(x, y)
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


def acquisition_point(context, row, column):
    capture = context.capture
    corrected = context.calibrated
    profile = corrected.start
    start_weight, end_weight = corrected.reference_weights(row, column)
    result = {
        "row": row,
        "source_line": row,
        "encoder_tick": int(capture.ticks[row]),
        "time_s": float(capture.seconds[row]),
        "station_mm": float(capture.stations[row]),
        "sensor_code": int(capture.raw[row, column]),
        "reference_basis": corrected.basis_name(row, column),
        "transition_fraction": float(corrected.transition[row]),
        "start_reference_weight": start_weight,
        "end_reference_weight": end_weight,
        "start_dark_code": float(profile.dark[column]),
        "start_white_code": float(profile.white[column]),
        "start_reference_faults": list(profile.fault_names(column)),
        "sample_saturated": bool(corrected.clipped[row, column]),
        "valid": bool(corrected.valid[row, column]),
        "reflectance": safe_value(corrected.values[row, column]),
    }
    if corrected.paired:
        end = corrected.end
        result.update(
            end_dark_code=float(end.dark[column]),
            end_white_code=float(end.white[column]),
            end_reference_faults=list(end.fault_names(column)),
            start_reference_time_s=corrected.reference_seconds[0],
            end_reference_time_s=corrected.reference_seconds[1],
        )
    else:
        result["reference_time_s"] = corrected.reference_seconds[0]
    return result


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

