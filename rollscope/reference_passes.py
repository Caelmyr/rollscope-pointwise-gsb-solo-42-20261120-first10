import numpy as np

from .calibration import safe_value, scale_summary


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


def column_drift(samples, fitted, usable, maximum_saturation):
    clipped = samples >= maximum_saturation
    valid = usable[None, :] & ~clipped
    differences = samples.astype(np.float32) - fitted[None, :]
    recent = max(1, len(samples) // 3)
    early_sum = np.where(valid[:recent], differences[:recent], 0).sum(axis=0)
    late_sum = np.where(valid[-recent:], differences[-recent:], 0).sum(axis=0)
    n_early = valid[:recent].sum(axis=0)
    n_late = valid[-recent:].sum(axis=0)
    estimated_before = np.full(fitted.shape, np.nan, dtype=np.float32)
    estimated_after = np.full(fitted.shape, np.nan, dtype=np.float32)
    np.divide(early_sum, n_early, out=estimated_before, where=n_early > 0)
    np.divide(late_sum, n_late, out=estimated_after, where=n_late > 0)
    if len(samples) < 2:
        return estimated_after * np.nan
    return estimated_after - estimated_before


def report_reference_passes(capture, profile, settings):
    sources = [
        ("dark", capture.dark, profile.dark),
        ("white", capture.white, profile.white),
    ]
    tables = {}
    drifts = {}
    for name, scans, fitted in sources:
        tables[name] = list(
            pass_measurements(
                scans, fitted, profile.usable, settings.saturation_code, settings.maximum_reference_noise
            )
        )
        drifts[name] = column_drift(scans, fitted, profile.usable, settings.saturation_code)

    curves = [
        {
            "transverse_mm": float(transverse),
            "dark_drift_code": safe_value(drifts["dark"][index]),
            "white_drift_code": safe_value(drifts["white"][index]),
        }
        for index, transverse in enumerate(capture.transverse)
    ]
    statistics = {
        "dark_paired_drift_code": scale_summary(drifts["dark"]),
        "white_paired_drift_code": scale_summary(drifts["white"]),
        "dark_reference_passes": len(tables["dark"]),
        "white_reference_passes": len(tables["white"]),
    }
    return statistics, tables, curves
