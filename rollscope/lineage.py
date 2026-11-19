import numpy as np

from .calibration import safe_value
from .grid import Support, index_near


def source_details(context, group, weight, column, contributes):
    capture = context.capture
    corrected = context.calibrated
    indices = np.array(list(context.grid.groups.sources(group)), dtype=np.int64)
    eligible = corrected.valid[indices, column]
    number = int(np.count_nonzero(eligible))
    effective_share = weight / number if number and contributes else 0.0
    details = []
    for offset, source in enumerate(indices):
        valid = eligible[offset]
        details.append(
            {
                "source_line": int(source),
                "capture_time_s": float(capture.seconds[source]),
                "encoder_tick": int(capture.ticks[source]),
                "sensor_code": int(capture.raw[source, column]),
                "reflectance": safe_value(corrected.values[source, column]),
                "group_weight": weight,
                "effective_weight": effective_share if valid else 0.0,
                "usable": bool(valid),
                "saturated": bool(corrected.clipped[source, column]),
            }
        )
    return details


def station_lineage(context, row, column):
    grid = context.grid
    support = Support(grid.plan.support[row])
    possible = []
    before = int(grid.plan.left[row])
    after = int(grid.plan.right[row])
    fraction = float(grid.plan.right_weight[row])
    contributes = bool(grid.valid[row, column])
    if support == Support.GAP:
        possible.extend(source_details(context, before, 0.0, column, False))
        if after != before:
            possible.extend(source_details(context, after, 0.0, column, False))
    else:
        possible.extend(source_details(context, before, 1.0 - fraction, column, contributes))
        if support == Support.BLEND:
            possible.extend(source_details(context, after, fraction, column, contributes))
    return possible


def acquisition_context(context, source_line, column, radius):
    capture = context.capture
    corrected = context.calibrated
    start = max(0, source_line - radius)
    stop = min(capture.rows, source_line + radius + 1)
    stations = capture.stations
    result = []
    for index in range(start, stop):
        station = float(stations[index])
        output_row = index_near(context.grid.stations, station)
        result.append(
            {
                "source_line": index,
                "station_mm": station,
                "tick": int(capture.ticks[index]),
                "time_s": float(capture.seconds[index]),
                "sensor_code": int(capture.raw[index, column]),
                "reflectance": safe_value(corrected.values[index, column]),
                "nearest_station_row": output_row,
                "selected_source": index == source_line,
            }
        )
    return result


def detailed_lineage(context, row, column, layer):
    if layer in ("sensor", "flat"):
        acquired = acquisition_context(context, row, column, 4)
        return {
            "kind": "acquisition",
            "line_count": len(acquired),
            "segments": [{"kind": "neighboring_scan_lines", "rows": acquired}],
        }
    records = station_lineage(context, row, column)
    support = Support(context.grid.plan.support[row])
    kind = "bracketing_lines" if support == Support.GAP else "source_fragments"
    total = sum(
        item["effective_weight"] * item["reflectance"]
        for item in records
        if item["effective_weight"] > 0
    )
    valid_weights = sum(item["effective_weight"] for item in records)
    report = {
        "kind": "station",
        "line_count": len(records),
        "composed_reflectance": total if valid_weights > 0 else None,
        "sum_of_weights": valid_weights,
        "segments": [{"kind": kind, "rows": records}],
    }
    return report
