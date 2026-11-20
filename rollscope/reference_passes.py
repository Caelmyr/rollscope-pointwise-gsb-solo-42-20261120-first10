import numpy as np

from .calibration import ReferenceBasis, safe_value, scale_summary


def pass_measurements(passes, fitted, good_columns, clipped_at, noise_limit):
    for index, samples in enumerate(passes):
        overflow = samples >= clipped_at
        support = good_columns & ~overflow
        difference = samples.astype(np.float32) - fitted
        available = difference[support]
        middle = low = high = absolute = None
        outlying = 0
        if len(available):
            low, middle, high = map(float, np.percentile(available, [10, 50, 90]))
            absolute = float(np.median(np.abs(available)))
            outlying = int(np.count_nonzero(np.abs(available) > noise_limit))
        yield {
            "pass": index,
            "usable_columns": int(np.count_nonzero(support)),
            "saturated_columns": int(np.count_nonzero(overflow)),
            "outlying_columns": outlying,
            "median_adjustment_code": middle,
            "p10_adjustment_code": low,
            "p90_adjustment_code": high,
            "median_absolute_change_code": absolute,
        }


def event_pass_tables(event_name, dark_passes, white_passes, profile, settings):
    tables = {}
    for kind, passes, fitted in (
        ("dark", dark_passes, profile.dark),
        ("white", white_passes, profile.white),
    ):
        rows = list(
            pass_measurements(
                passes, fitted, profile.usable, settings.saturation_code,
                settings.maximum_reference_noise,
            )
        )
        for row in rows:
            row["event"] = event_name
        tables[kind] = rows
    return tables


def report_reference_passes(capture, calibrated, settings):
    dark_sets, white_sets = capture.reference_sets()
    events = ("start", "end") if calibrated.paired else ("start",)
    profiles = (calibrated.start, calibrated.end) if calibrated.paired else (calibrated.start,)

    tables = {"dark": [], "white": []}
    for event, profile, dark_passes, white_passes in zip(
        events, profiles, dark_sets, white_sets
    ):
        event_tables = event_pass_tables(
            event, dark_passes, white_passes, profile, settings
        )
        tables["dark"].extend(event_tables["dark"])
        tables["white"].extend(event_tables["white"])

    statistics = {
        "reference_mode": calibrated.mode,
        "reference_seconds": list(calibrated.reference_seconds),
        "trust_reach_fraction": float(calibrated.trust_reach_fraction),
        "dark_reference_passes": sum(map(len, dark_sets)),
        "white_reference_passes": sum(map(len, white_sets)),
        "reference_events": len(events),
        "basis_cells": calibrated.basis_counts(),
    }

    start = calibrated.start
    if calibrated.paired:
        end = calibrated.end
        dark_drift = end.dark - start.dark
        white_drift = end.white - start.white
        statistics["dark_event_drift_code"] = scale_summary(dark_drift)
        statistics["white_event_drift_code"] = scale_summary(white_drift)
        statistics["response_event_drift_code"] = scale_summary(end.response - start.response)
        statistics["start_rejected_columns"] = int(np.count_nonzero(~start.usable))
        statistics["end_rejected_columns"] = int(np.count_nonzero(~end.usable))
    else:
        dark_drift = np.full_like(start.dark, np.nan)
        white_drift = np.full_like(start.white, np.nan)

    curves = [
        {
            "transverse_mm": float(transverse),
            "dark_drift_code": safe_value(dark_drift[index]),
            "white_drift_code": safe_value(white_drift[index]),
            "start_response_code": float(start.response[index]),
            "end_response_code": (
                float(calibrated.end.response[index]) if calibrated.paired else None
            ),
            "start_usable": bool(start.usable[index]),
            "end_usable": bool(calibrated.end.usable[index]) if calibrated.paired else None,
        }
        for index, transverse in enumerate(capture.transverse)
    ]
    return statistics, tables, curves


def basis_series(calibrated, stations):
    """Per-station counts of which reference basis each line relied on."""
    counts = np.zeros((len(stations), len(ReferenceBasis)), dtype=np.int64)
    for basis in ReferenceBasis:
        counts[:, basis.value] = np.count_nonzero(
            calibrated.basis == basis.value, axis=1
        )
    return [
        {
            "station_mm": float(stations[index]),
            "transition_fraction": float(calibrated.transition[index]),
            **{
                basis.name.lower(): int(counts[index, basis.value])
                for basis in ReferenceBasis
            },
        }
        for index in range(len(stations))
    ]
