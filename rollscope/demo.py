from dataclasses import dataclass

import numpy as np

from .domain import Capture, CaptureMetadata, Geometry, UserInputError


@dataclass(frozen=True)
class DemoProfile:
    material: str
    name: str
    description: str
    rows: int
    columns: int
    encoder_pitch: float
    transverse_pitch: float
    station_origin: float
    seed: int


PROFILES = {
    "cold-rolled": DemoProfile(
        "Cold-rolled steel",
        "Coil R24-071 / side A",
        "Synthetic line scan with two encoder interruptions, repeated ticks, edge wander, "
        "a rolling streak and a transverse join. It is a simulated inspection capture.",
        1140,
        320,
        0.18,
        0.28,
        1200,
        24071,
    ),
    "coated": DemoProfile(
        "Coated aluminum",
        "Coil A08-016 / uncoated face",
        "Synthetic wider-pitch line scan with a narrow reflective stripe, two joins, "
        "a longer transport interruption and darker material than the steel example.",
        920,
        240,
        0.22,
        0.34,
        0,
        8016,
    ),
}


def encoder_program(profile, rng):
    increments = np.full(profile.rows, 4, dtype=np.int64)
    increments[0] = 0
    offsets = np.arange(29, profile.rows - 1, 37)
    increments[offsets] = rng.choice([3, 5], size=len(offsets))
    repeated = np.arange(64, profile.rows - 1, 127)
    increments[repeated] = 0
    interruptions = (profile.rows // 3, int(profile.rows * 0.73))
    first_jump = 114 if profile.material == "Coated aluminum" else 65
    increments[interruptions[0]] = first_jump
    increments[interruptions[1]] = 43

    ticks = np.cumsum(increments)
    ticks += 35 if profile.station_origin else -20
    times = 0.004 + rng.uniform(-0.00063, 0.00063, profile.rows)
    times[0] = 0
    times[interruptions[0]] += 0.108
    times[interruptions[1]] += 0.038
    times[int(profile.rows * 0.9)] += 0.024
    return ticks, np.cumsum(times)


def references(profile, rng):
    position = np.arange(profile.columns)
    black_level = 820 + 33 * np.sin(position / 32) + 22 * np.cos(position / 4.5)
    sensitivity = 39600 * (1 + 0.1 * np.cos((position - 20) / 69.0))
    sensitivity += 1600 * np.sin(position / 18.5)
    sensitivity[[22, profile.columns - 29]] = [520, 270]
    if profile.material == "Coated aluminum":
        sensitivity *= 0.96
    dark = black_level + rng.normal(0, 12, size=(9, profile.columns))
    white = black_level + sensitivity + rng.normal(0, 35, size=(11, profile.columns))
    noise_pattern = np.array([-600, 340, -530, 590, -580, 490, 630, -480, 380, 570, -320])
    white[:, profile.columns // 5] += noise_pattern
    return (
        black_level,
        sensitivity,
        np.rint(dark).clip(0, 65535).astype(np.uint16),
        np.rint(white).clip(0, 65535).astype(np.uint16),
    )


def strip_bounds(stations, width):
    relative = stations - stations[0]
    wander = 4.7 * np.sin(relative / 117) + 1.8 * np.cos(relative / 47)
    taper = 2.6 * np.sin(relative / 154.0)
    left = width * 0.111 + wander + taper
    right = width * 0.891 + wander - taper
    left += 3.5 * np.exp(-0.5 * ((relative - 459) / 23) ** 2)
    return left, right


def material_pattern(profile, stations, rng):
    y = (stations - stations[0])[:, None]
    columns = np.arange(profile.columns)[None, :]
    x = columns * profile.transverse_pitch
    left, right = strip_bounds(stations, profile.columns)
    left = left[:, None]
    right = right[:, None]
    progress = y[:, 0] / y[-1, 0]
    width = right - left
    across = (columns - left) / width
    support = np.clip(0.6 + np.minimum(columns - left, right - columns), 0, 1)

    exposure = 0.47 if profile.material == "Coated aluminum" else 0.574
    texture = (
        0.022 * np.sin(2 * np.pi * (x / 1.24 + y / 10.7))
        + 0.012 * np.cos(2 * np.pi * (x / 3.87 - y / 12.9))
        + 0.0051 * np.sin(2 * np.pi * (x / 0.88 + y / 2.95))
    )
    illumination = 0.033 * np.cos(across * 2.5 - 0.7) - 0.038 * progress[:, None]
    reflectance = exposure + texture + illumination
    first_streak = 0.107 * np.exp(-0.5 * ((x - x.max() * 0.401) / 0.49) ** 2)
    streak_gate = (progress > 0.149) & (progress < 0.66)
    reflectance -= first_streak * streak_gate[:, None]
    if profile.material == "Coated aluminum":
        second = 0.078 * np.exp(-0.5 * ((x - x.max() * 0.562) / 0.43) ** 2)
        reflectance += second * (progress > 0.37)[:, None]

    for center_y, fraction, spread_y, spread_x, amplitude in spot_program(profile):
        center_x = x.max() * fraction
        distance = ((y - center_y) / spread_y) ** 2 + ((x - center_x) / spread_x) ** 2
        reflectance += amplitude * np.exp(-0.5 * distance)

    offsets = np.arange(30, y[-1, 0] - 25, 60, dtype=np.float64)
    offsets[1::3] += 1.65
    marks = np.zeros_like(y)
    for offset in offsets:
        marks += 0.369 * np.exp(-0.5 * ((y - offset) / 1.06) ** 2)
    mark_shape = np.exp(-0.5 * ((across - 0.81) / 0.11) ** 6)
    reflectance -= marks * mark_shape

    join_positions = (0.532,) if profile.material == "Cold-rolled steel" else (0.288, 0.68)
    for join_position in join_positions:
        join_y = y[-1, 0] * join_position
        adjustment = 0.38 * np.sin(4 * across)
        bright = np.exp(-0.5 * ((y - join_y + adjustment) / 0.95) ** 2)
        dark = np.exp(-0.5 * ((y - join_y - 2.1 + adjustment) / 1.24) ** 2)
        reflectance += 0.243 * bright - 0.131 * dark

    sensor_noise = rng.normal(0, 0.0029, size=(profile.rows, profile.columns))
    backdrop = 0.084 + 0.017 * np.cos(columns / 31) + 0.007 * np.sin(y / 71)
    return backdrop * (1 - support) + reflectance * support + sensor_noise


def spot_program(profile):
    if profile.material == "Coated aluminum":
        return (
            (111, 0.36, 2.0, 0.64, 0.36),
            (234, 0.63, 3.0, 1.2, -0.39),
            (480, 0.28, 3.8, 1.34, -0.34),
            (610, 0.42, 1.7, 1.0, 0.35),
        )
    return (
        (118, 0.55, 2.4, 1.05, -0.37),
        (208, 0.37, 2.2, 0.68, 0.32),
        (343, 0.62, 1.4, 0.83, -0.33),
        (402, 0.28, 4.4, 1.58, -0.4),
        (689, 0.66, 1.5, 0.82, 0.34),
        (730, 0.51, 2.8, 3.2, -0.27),
    )


def build_demo(name):
    try:
        profile = PROFILES[name]
    except KeyError as error:
        raise UserInputError("Choose one of the available synthetic captures.") from error
    rng = np.random.default_rng(profile.seed)
    geometry = Geometry(
        profile.encoder_pitch,
        profile.transverse_pitch,
        profile.station_origin,
        0.0,
    )
    ticks, seconds = encoder_program(profile, rng)
    dark_level, sensitivity, dark, white = references(profile, rng)
    reflectance = material_pattern(profile, geometry.station(ticks), rng)
    raw = np.rint(dark_level + sensitivity * reflectance)
    raw[profile.rows // 2, profile.columns // 2 : profile.columns // 2 + 4] = 65535
    raw = raw.clip(0, 65535).astype(np.uint16)
    metadata = CaptureMetadata(
        profile.name, profile.material, profile.description, True, geometry
    )
    return Capture.build(raw, ticks, seconds, dark, white, metadata)


def available_demos():
    return [
        {
            "id": key,
            "name": profile.name,
            "material": profile.material,
            "description": profile.description,
        }
        for key, profile in PROFILES.items()
    ]
