import numpy as np

from .calibration import safe_value
from .grid import Support
from .lineage import detailed_lineage
from .presentation import extrema_indices
from .raster import ACQUISITION_LAYERS, Layer, position_mapping


def point_measurement(context, spec, x, y):
    mapping = position_mapping(context, spec)
    row, column = mapping.cell(x, y)
    if mapping.block_mode:
        block = mapping.block_at_pixel(x, y)
        result = block_measurement(context, spec, mapping, block)
        result["coordinate"] = mapping.as_dict()
        result["layer"] = spec.layer.value
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


def _support_counts(facts, block_row, block_col):
    return {
        "native_cells": int(facts.native_cells[block_row, block_col]),
        "blend_cells": int(facts.blend_cells[block_row, block_col]),
        "gap_cells": int(facts.gap_cells[block_row, block_col]),
        "invalid_cells": int(facts.bad_cells[block_row, block_col]),
        "valid_cells": int(facts.valid_cells[block_row, block_col]),
        "total_cells": int(facts.total_cells[block_row, block_col]),
    }


def block_measurement(context, spec, mapping, block):
    """Summarize the source cells collapsed into one low-magnification block."""
    block_row, block_col = block
    facts = context.block_facts(spec.layer, mapping.block_grid)
    grid = context.grid
    row_slice, column_slice = mapping.block_source_bounds(block_row, block_col)
    stations = grid.stations
    transverse = context.capture.transverse

    support = _support_counts(facts, block_row, block_col)
    total = max(1, support["total_cells"])
    partial_evidence = []
    if support["gap_cells"]:
        partial_evidence.append({
            "kind": "encoder_gap",
            "cells": support["gap_cells"],
            "fraction": round(support["gap_cells"] / total, 4),
        })
    if support["invalid_cells"]:
        partial_evidence.append({
            "kind": "invalid_sample",
            "cells": support["invalid_cells"],
            "fraction": round(support["invalid_cells"] / total, 4),
        })
    dark_anomalies = int(facts.dark_anomaly_cells[block_row, block_col])
    bright_anomalies = int(facts.bright_anomaly_cells[block_row, block_col])
    if dark_anomalies:
        partial_evidence.append({
            "kind": "dark_anomaly",
            "cells": dark_anomalies,
            "fraction": round(dark_anomalies / total, 4),
        })
    if bright_anomalies:
        partial_evidence.append({
            "kind": "bright_anomaly",
            "cells": bright_anomalies,
            "fraction": round(bright_anomalies / total, 4),
        })
    if spec.layer == Layer.SPOTS:
        spot_ids = _spots_in_block(context, row_slice, column_slice)
        if spot_ids:
            partial_evidence.append({"kind": "spot_footprint", "spot_ids": spot_ids})

    result = {
        "probe_kind": "block",
        "block_row": int(block_row),
        "block_column": int(block_col),
        "source_row_start": int(row_slice.start),
        "source_row_end": int(row_slice.stop - 1),
        "source_column_start": int(column_slice.start),
        "source_column_end": int(column_slice.stop - 1),
        "station_start_mm": float(stations[row_slice.start]),
        "station_end_mm": float(stations[row_slice.stop - 1]),
        "transverse_start_mm": float(transverse[column_slice.start]),
        "transverse_end_mm": float(transverse[column_slice.stop - 1]),
        "median_reflectance": safe_value(facts.median[block_row, block_col]),
        "minimum_reflectance": safe_value(facts.minimum[block_row, block_col]),
        "maximum_reflectance": safe_value(facts.maximum[block_row, block_col]),
        "valid_fraction": round(float(facts.valid_fraction()[block_row, block_col]), 4),
        "support": support,
        "partial_evidence": partial_evidence,
        "region": {
            "station_start_mm": float(stations[row_slice.start]),
            "station_end_mm": float(stations[row_slice.stop - 1]),
            "transverse_start_mm": float(transverse[column_slice.start]),
            "transverse_end_mm": float(transverse[column_slice.stop - 1]),
        },
    }
    if spec.layer in (Layer.SPOTS, Layer.STREAKS):
        result["contrast_minimum"] = safe_value(facts.contrast_min[block_row, block_col])
        result["contrast_maximum"] = safe_value(facts.contrast_max[block_row, block_col])
    return result


def _spots_in_block(context, row_slice, column_slice):
    labels = context.spots.labels[row_slice, column_slice]
    present = np.unique(labels[labels > 0])
    return [int(identifier) for identifier in present]


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
