from dataclasses import asdict, dataclass

import numpy as np

from .calibration import scale_summary


MAX_EVALUATED_TILES = 720
MAX_REPORTED_TILES = 240
ANGLE_EDGES = np.linspace(-90.0, 90.0, 19)


@dataclass(frozen=True)
class TextureTile:
    id: int
    station_mm: float
    transverse_mm: float
    from_mm: float
    to_mm: float
    left_mm: float
    right_mm: float
    angle_deg: float | None
    directional_strength: float
    band_rms: float
    wavelength_mm: float | None

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SpectrumAxes:
    radius: np.ndarray
    angle_deg: np.ndarray
    angle_bins: np.ndarray
    spectral_bins: np.ndarray
    selected: np.ndarray
    wave_edges: np.ndarray
    unit_vector: np.ndarray


@dataclass(frozen=True)
class SpectrumMeasurement:
    power: float
    direction_vector: complex
    dominant_wave_mm: float | None
    direction_power: np.ndarray
    wavelength_power: np.ndarray

    @property
    def direction(self):
        if self.strength < 0.045:
            return None
        return float(np.degrees(np.angle(self.direction_vector)) / 2)

    @property
    def strength(self):
        return 0.0 if self.power <= 1e-12 else float(abs(self.direction_vector) / self.power)


@dataclass(frozen=True)
class TextureReport:
    tiles: tuple
    total_positions: int
    eligible_positions: int
    evaluated_positions: int
    angle_power: np.ndarray
    wavelength_power: np.ndarray
    wave_edges: np.ndarray
    combined_direction: float | None
    combined_strength: float

    def summary(self):
        return {
            "candidate_tiles": self.total_positions,
            "eligible_tiles": self.eligible_positions,
            "evaluated_tiles": self.evaluated_positions,
            "displayed_tiles": len(self.tiles),
            "stripe_direction_deg": self.combined_direction,
            "directional_strength": self.combined_strength,
            "band_rms": scale_summary([tile.band_rms for tile in self.tiles]),
        }

    def table(self):
        return [tile.as_dict() for tile in self.tiles]

    def orientations(self):
        return [
            {
                "from_deg": float(low),
                "to_deg": float(high),
                "fraction": float(fraction),
            }
            for low, high, fraction in zip(ANGLE_EDGES[:-1], ANGLE_EDGES[1:], self.angle_power)
        ]

    def wavelength_bands(self):
        return [
            {"from_mm": float(low), "to_mm": float(high), "fraction": float(fraction)}
            for low, high, fraction in zip(
                self.wave_edges[:-1], self.wave_edges[1:], self.wavelength_power
            )
        ]


def powers_normalized(counts):
    total = float(np.sum(counts))
    return counts / total if total > 1e-12 else np.zeros_like(counts)


def start_cells(length, size, stride):
    if length < size:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(0, length - size + 1, stride)
    final_start = length - size
    if starts[-1] == final_start:
        return starts
    return np.append(starts, final_start)


def tile_positions(support, size, overlap):
    stride = max(1, int(round(size * (1 - overlap))))
    rows = start_cells(support.shape[0], size, stride)
    columns = start_cells(support.shape[1], size, stride)
    if not rows.size or not columns.size:
        return np.zeros((0, 2), dtype=np.int64), len(rows) * len(columns)
    integral = np.pad(np.cumsum(np.cumsum(~support, axis=0), axis=1), ((1, 0), (1, 0)))
    row, col = np.meshgrid(rows, columns, indexing="ij")
    invalid = (
        integral[row + size, col + size]
        - integral[row, col + size]
        - integral[row + size, col]
        + integral[row, col]
    )
    eligible = invalid == 0
    return np.column_stack((row[eligible], col[eligible])), len(rows) * len(columns)


def balanced_sampling(items, limit):
    if len(items) <= limit:
        return items
    return items[np.linspace(0, len(items) - 1, limit, dtype=np.int64)]


def spectral_axes(size, row_pitch, column_pitch, shortest, longest):
    longitudinal = np.fft.fftfreq(size, d=row_pitch)[:, None]
    transverse = np.fft.fftfreq(size, d=column_pitch)[None, :]
    magnitude = np.hypot(longitudinal, transverse)
    selected = (magnitude >= 1 / longest) & (magnitude <= 1 / shortest)
    radius = magnitude[selected]
    orientation = np.degrees(np.arctan2(longitudinal, transverse))
    angle_deg = (orientation + 90) % 180 - 90
    angle_deg = angle_deg[selected]
    angles = np.searchsorted(ANGLE_EDGES, angle_deg, side="right") - 1
    wave_edges = np.geomspace(shortest, longest, 13)
    wavelengths = np.reciprocal(radius)
    wave_bins = np.clip(np.searchsorted(wave_edges, wavelengths, side="right") - 1, 0, 11)
    unit = np.exp(2j * np.radians(angle_deg))
    return SpectrumAxes(radius, angle_deg, angles, wave_bins, selected, wave_edges, unit)


def taper_and_detrend(patch, linear, ramp_energy, window):
    scaled_patch = patch.astype(np.float64)
    center = np.mean(scaled_patch)
    horizontal = np.dot(np.sum(scaled_patch, axis=0), linear) / ramp_energy
    vertical = np.dot(np.sum(scaled_patch, axis=1), linear) / ramp_energy
    plane = center + horizontal * linear[None, :] + vertical * linear[:, None]
    return window * (scaled_patch - plane)


def tile_spectrum(patch, axes, linear, ramp_energy, window, window_energy):
    tapered = taper_and_detrend(patch, linear, ramp_energy, window)
    image = np.fft.fft2(tapered)
    power = (image.real**2 + image.imag**2)[axes.selected] / window_energy
    measured = float(np.sum(power))
    angle_strength = np.bincount(axes.angle_bins, weights=power, minlength=18)
    wavelength_strength = np.bincount(axes.spectral_bins, weights=power, minlength=12)
    vector = np.dot(power, axes.unit_vector)
    midpoint = None
    if measured > 1e-12:
        strongest = int(np.argmax(wavelength_strength))
        midpoint = float(np.sqrt(axes.wave_edges[strongest] * axes.wave_edges[strongest + 1]))
    return SpectrumMeasurement(measured, complex(vector), midpoint, angle_strength, wavelength_strength)


def tile_evidence(grid, row, column, size, texture, identifier):
    stations = grid.stations[[row, row + size - 1]]
    columns = grid.transverse[[column, column + size - 1]]
    return TextureTile(
        identifier,
        float(np.mean(stations)),
        float(np.mean(columns)),
        float(stations[0]),
        float(stations[-1]),
        float(columns[0]),
        float(columns[-1]),
        texture.direction,
        texture.strength,
        float(np.sqrt(texture.power)),
        texture.dominant_wave_mm,
    )


def measure_texture(grid, edges, settings):
    size = settings.tile_size
    support = edges.inner_mask(2)
    eligible, total_positions = tile_positions(support, size, settings.overlap_fraction)
    chosen = balanced_sampling(eligible, MAX_EVALUATED_TILES)
    linear = np.arange(size, dtype=np.float64) - (size - 1) / 2
    energy = np.dot(linear, linear) * size
    hann = np.hanning(size)
    window = hann[:, None] * hann[None, :]
    window_energy = np.sum(window**2) * size * size
    axes = spectral_axes(
        size,
        grid.row_pitch_mm,
        grid.column_pitch_mm,
        settings.minimum_wave_mm,
        settings.maximum_wave_mm,
    )
    tiles = []
    direction_total = np.zeros(18, dtype=float)
    wavelength_total = np.zeros(12, dtype=float)
    vector = 0j
    power = 0.0
    for identifier, (row, column) in enumerate(chosen, 1):
        patch = grid.values[row : row + size, column : column + size]
        outcome = tile_spectrum(patch, axes, linear, energy, window, window_energy)
        direction_total += outcome.direction_power
        wavelength_total += outcome.wavelength_power
        power += outcome.power
        vector += outcome.direction_vector
        tiles.append(tile_evidence(grid, row, column, size, outcome, identifier))
    combined = SpectrumMeasurement(power, vector, None, direction_total, wavelength_total)
    return TextureReport(
        tuple(balanced_sampling(np.asarray(tiles, dtype=object), MAX_REPORTED_TILES)),
        total_positions,
        len(eligible),
        len(chosen),
        powers_normalized(direction_total),
        powers_normalized(wavelength_total),
        axes.wave_edges,
        combined.direction,
        combined.strength,
    )
