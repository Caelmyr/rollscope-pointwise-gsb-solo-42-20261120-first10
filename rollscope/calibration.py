from dataclasses import dataclass
from enum import IntFlag

import numpy as np


class ReferenceFault(IntFlag):
    LOW_RESPONSE = 1
    UNSTABLE_DARK = 2
    UNSTABLE_WHITE = 4
    SATURATED_REFERENCE = 8


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


@dataclass(frozen=True)
class CalibratedImage:
    values: np.ndarray
    valid: np.ndarray
    clipped: np.ndarray
    reference: ReferenceProfile

    def summary(self):
        valid_pixels = np.count_nonzero(self.valid)
        return {
            "valid_pixels": int(valid_pixels),
            "valid_fraction": float(valid_pixels / self.valid.size),
            "saturated_pixels": int(np.count_nonzero(self.clipped)),
            "reflectance": scale_summary(self.values),
            "reference": self.reference.summary(),
        }


def reference_profile(capture, settings):
    dark, dark_noise = median_and_spread(capture.dark)
    white, white_noise = median_and_spread(capture.white)
    response = white - dark
    flags = np.zeros(capture.columns, dtype=np.uint8)
    flags[response < settings.minimum_response] |= ReferenceFault.LOW_RESPONSE.value
    flags[dark_noise > settings.maximum_reference_noise] |= ReferenceFault.UNSTABLE_DARK.value
    flags[white_noise > settings.maximum_reference_noise] |= ReferenceFault.UNSTABLE_WHITE.value
    flags[(dark >= settings.saturation_code) | (white >= settings.saturation_code)] |= (
        ReferenceFault.SATURATED_REFERENCE.value
    )
    return ReferenceProfile(dark, white, response, dark_noise, white_noise, flags)


def calibrate(capture, settings):
    profile = reference_profile(capture, settings)
    clipped = capture.raw >= settings.saturation_code
    valid = profile.usable[None, :] & ~clipped
    corrected = np.full(capture.raw.shape, np.nan, dtype=np.float32)
    np.divide(
        capture.raw.astype(np.float32) - profile.dark[None, :],
        profile.response[None, :],
        where=valid,
        out=corrected,
    )
    return CalibratedImage(corrected, valid, clipped, profile)
