from dataclasses import dataclass
from enum import IntEnum, IntFlag

import numpy as np


class ReferenceFault(IntFlag):
    LOW_RESPONSE = 1
    UNSTABLE_DARK = 2
    UNSTABLE_WHITE = 4
    SATURATED_REFERENCE = 8


class ReferenceBasis(IntEnum):
    UNKNOWN = 0
    START = 1
    END = 2
    BLENDED = 3


def safe_value(value):
    return float(value) if np.isfinite(value) else None


def scale_summary(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    quantiles = np.percentile(finite, [0, 10, 50, 90, 100])
    keys = ("min", "p10", "median", "p90", "max")
    return dict(zip(keys, map(float, quantiles)))


def median_and_spread(reference):
    center = np.median(reference, axis=0).astype(np.float32)
    median_deviation = np.median(np.abs(reference - center), axis=0)
    spread = (median_deviation * 1.4826).astype(np.float32)
    return center, spread


@dataclass(frozen=True)
class ReferenceProfile:
    dark: np.ndarray
    white: np.ndarray
    response: np.ndarray
    dark_noise: np.ndarray
    white_noise: np.ndarray
    flags: np.ndarray

    @property
    def usable(self):
        return self.flags == 0

    def summary(self):
        flag_counts = {
            fault.name.lower(): int(np.count_nonzero(self.flags & fault.value))
            for fault in ReferenceFault
        }
        return {
            "usable_columns": int(np.count_nonzero(self.usable)),
            "rejected_columns": int(np.count_nonzero(~self.usable)),
            "fault_counts": flag_counts,
            "usable_response": scale_summary(self.response[self.usable]),
            "dark_code": scale_summary(self.dark),
        }

    def column_records(self, transverse):
        return [
            {
                "column": index,
                "transverse_mm": float(transverse[index]),
                "dark_code": float(self.dark[index]),
                "white_code": float(self.white[index]),
                "response_code": float(self.response[index]),
                "dark_noise_code": float(self.dark_noise[index]),
                "white_noise_code": float(self.white_noise[index]),
                "valid": bool(self.usable[index]),
                "faults": list(self.fault_names(index)),
            }
            for index in range(len(self.flags))
        ]

    def fault_names(self, column):
        for fault in ReferenceFault:
            if self.flags[column] & fault.value:
                yield fault.name.lower()


def endpoint_profile(dark_passes, white_passes, settings):
    dark, dark_noise = median_and_spread(dark_passes)
    white, white_noise = median_and_spread(white_passes)
    response = white - dark
    flags = np.zeros(len(response), dtype=np.uint8)
    flags[response < settings.minimum_response] |= ReferenceFault.LOW_RESPONSE.value
    flags[dark_noise > settings.maximum_reference_noise] |= ReferenceFault.UNSTABLE_DARK.value
    flags[white_noise > settings.maximum_reference_noise] |= ReferenceFault.UNSTABLE_WHITE.value
    flags[(dark >= settings.saturation_code) | (white >= settings.saturation_code)] |= (
        ReferenceFault.SATURATED_REFERENCE.value
    )
    return ReferenceProfile(dark, white, response, dark_noise, white_noise, flags)


def reference_profile(capture, settings):
    """Profile of the start-of-roll reference event (the legacy single set)."""
    return endpoint_profile(capture.dark, capture.white, settings)


@dataclass(frozen=True)
class CalibratedImage:
    values: np.ndarray
    valid: np.ndarray
    clipped: np.ndarray
    start: ReferenceProfile
    basis: np.ndarray
    transition: np.ndarray
    reference_seconds: tuple
    trust_reach_fraction: float
    end: ReferenceProfile | None = None

    @property
    def reference(self):
        return self.start

    @property
    def paired(self):
        return self.end is not None

    @property
    def mode(self):
        return "paired" if self.paired else "single"

    def reference_weights(self, row, column):
        """Return (start_weight, end_weight) actually applied to one pixel."""
        kind = ReferenceBasis(int(self.basis[row, column]))
        if kind == ReferenceBasis.UNKNOWN:
            return 0.0, 0.0
        if kind == ReferenceBasis.START:
            return 1.0, 0.0
        if kind == ReferenceBasis.END:
            return 0.0, 1.0
        fraction = float(self.transition[row])
        return 1.0 - fraction, fraction

    def basis_name(self, row, column):
        return ReferenceBasis(int(self.basis[row, column])).name.lower()

    def basis_counts(self):
        counts = np.bincount(self.basis.ravel(), minlength=len(ReferenceBasis))
        return {basis.name.lower(): int(counts[basis.value]) for basis in ReferenceBasis}

    def endpoint_summary(self):
        result = {
            "mode": self.mode,
            "reference_events": 2 if self.paired else 1,
            "reference_seconds": list(self.reference_seconds),
            "trust_reach_fraction": float(self.trust_reach_fraction),
            "basis_cells": self.basis_counts(),
            "start": self.start.summary(),
        }
        if self.paired:
            result["end"] = self.end.summary()
        return result

    def paired_column_records(self, transverse):
        start, end = self.start, self.end
        for index in range(len(self.start.flags)):
            record = {
                "column": index,
                "transverse_mm": float(transverse[index]),
                "start_dark_code": float(start.dark[index]),
                "start_white_code": float(start.white[index]),
                "start_response_code": float(start.response[index]),
                "start_valid": bool(start.usable[index]),
                "start_faults": list(start.fault_names(index)),
            }
            if self.paired:
                record.update(
                    end_dark_code=float(end.dark[index]),
                    end_white_code=float(end.white[index]),
                    end_response_code=float(end.response[index]),
                    end_valid=bool(end.usable[index]),
                    end_faults=list(end.fault_names(index)),
                    response_drift_code=float(end.response[index] - start.response[index]),
                    white_drift_code=float(end.white[index] - start.white[index]),
                    dark_drift_code=float(end.dark[index] - start.dark[index]),
                )
            yield record

    def summary(self):
        valid_pixels = np.count_nonzero(self.valid)
        return {
            "valid_pixels": int(valid_pixels),
            "valid_fraction": float(valid_pixels / self.valid.size),
            "saturated_pixels": int(np.count_nonzero(self.clipped)),
            "reflectance": scale_summary(self.values),
            "reference": self.start.summary(),
            "reference_calibration": self.endpoint_summary(),
        }


def station_transition(capture):
    stations = capture.stations
    span = float(stations[-1] - stations[0])
    if span <= 0:
        return np.zeros(len(stations), dtype=np.float32)
    fraction = (stations - stations[0]) / span
    return np.clip(fraction, 0.0, 1.0).astype(np.float32)


def paired_calibration(capture, start, end, settings, clipped):
    rows, columns = capture.raw.shape
    fraction = station_transition(capture)
    blend = fraction[:, None]
    start_ok = start.usable
    end_ok = end.usable
    both = start_ok & end_ok
    reach = settings.reference_trust_reach_fraction

    inside_start = (fraction <= reach)[:, None]
    inside_end = (fraction >= 1.0 - reach)[:, None]
    trusted = np.broadcast_to(both, (rows, columns)).copy()
    trusted |= start_ok[None, :] & inside_start
    trusted |= end_ok[None, :] & inside_end

    basis = np.full((rows, columns), ReferenceBasis.UNKNOWN.value, dtype=np.uint8)
    basis[np.broadcast_to(start_ok, (rows, columns))] = ReferenceBasis.START.value
    basis[np.broadcast_to(end_ok, (rows, columns))] = ReferenceBasis.END.value
    basis[np.broadcast_to(both, (rows, columns))] = ReferenceBasis.BLENDED.value
    basis[~trusted] = ReferenceBasis.UNKNOWN.value

    effective_dark = np.full((rows, columns), np.nan, dtype=np.float32)
    effective_white = np.full((rows, columns), np.nan, dtype=np.float32)
    dark_0 = start.dark[None, :]
    dark_1 = end.dark[None, :]
    white_0 = start.white[None, :]
    white_1 = end.white[None, :]
    blended_dark = (1.0 - blend) * dark_0 + blend * dark_1
    blended_white = (1.0 - blend) * white_0 + blend * white_1
    np.copyto(effective_dark, blended_dark, where=both[None, :])
    np.copyto(effective_white, blended_white, where=both[None, :])
    start_only = start_ok & ~end_ok
    end_only = end_ok & ~start_ok
    np.copyto(effective_dark, np.broadcast_to(dark_0, effective_dark.shape),
              where=start_only[None, :] & inside_start)
    np.copyto(effective_white, np.broadcast_to(white_0, effective_white.shape),
              where=start_only[None, :] & inside_start)
    np.copyto(effective_dark, np.broadcast_to(dark_1, effective_dark.shape),
              where=end_only[None, :] & inside_end)
    np.copyto(effective_white, np.broadcast_to(white_1, effective_white.shape),
              where=end_only[None, :] & inside_end)

    valid = trusted & ~clipped
    response = effective_white - effective_dark
    corrected = np.full((rows, columns), np.nan, dtype=np.float32)
    safe_response = valid & np.isfinite(response) & (response > 0)
    np.divide(
        capture.raw.astype(np.float32) - effective_dark,
        response,
        where=safe_response,
        out=corrected,
    )
    return corrected, valid, basis, fraction


def single_calibration(capture, profile, clipped):
    rows, columns = capture.raw.shape
    basis = np.full((rows, columns), ReferenceBasis.UNKNOWN.value, dtype=np.uint8)
    basis[:, profile.usable] = ReferenceBasis.START.value
    valid = np.broadcast_to(profile.usable, (rows, columns)) & ~clipped
    corrected = np.full((rows, columns), np.nan, dtype=np.float32)
    response = profile.response[None, :]
    safe_response = valid & np.isfinite(response) & (response > 0)
    np.divide(
        capture.raw.astype(np.float32) - profile.dark[None, :],
        response,
        where=safe_response,
        out=corrected,
    )
    return corrected, valid, basis, np.zeros(rows, dtype=np.float32)


def calibrate(capture, settings):
    start = endpoint_profile(capture.dark, capture.white, settings)
    clipped = capture.raw >= settings.saturation_code
    if capture.paired_references:
        end = endpoint_profile(capture.dark_end, capture.white_end, settings)
        values, valid, basis, transition = paired_calibration(
            capture, start, end, settings, clipped
        )
    else:
        end = None
        values, valid, basis, transition = single_calibration(capture, start, clipped)
    return CalibratedImage(
        values,
        valid,
        clipped,
        start,
        basis,
        transition,
        capture.reference_event_seconds(),
        float(settings.reference_trust_reach_fraction),
        end,
    )
