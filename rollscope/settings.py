from dataclasses import asdict, dataclass, fields

from .domain import UserInputError, expect_object, finite_number, reject_unknown, whole_number


def overlay(section, data, label):
    data = expect_object(data, label)
    reject_unknown(data, [field.name for field in fields(section)], label)
    return asdict(section) | data


@dataclass(frozen=True)
class CalibrationSettings:
    minimum_response: int = 1000
    maximum_reference_noise: float = 220.0
    saturation_code: int = 65520

    @classmethod
    def parse(cls, data):
        value = overlay(cls(), data, "Flat-field settings")
        return cls(
            minimum_response=whole_number(
                value["minimum_response"], "Minimum dark-to-white response", 1, 65000
            ),
            maximum_reference_noise=finite_number(
                value["maximum_reference_noise"], "Maximum reference variation", 1, 10000
            ),
            saturation_code=whole_number(
                value["saturation_code"], "Saturation code", 128, 65535
            ),
        )


@dataclass(frozen=True)
class GridSettings:
    pitch_mm: float | None = None
    gap_limit_mm: float | None = None

    @classmethod
    def parse(cls, data):
        value = overlay(cls(), data, "Station settings")
        pitch = value["pitch_mm"]
        limit = value["gap_limit_mm"]
        return cls(
            pitch_mm=(
                None if pitch is None else finite_number(pitch, "Station pitch", 0.001, 2000)
            ),
            gap_limit_mm=(
                None if limit is None else finite_number(limit, "Gap limit", 0.001, 5000)
            ),
        )


@dataclass(frozen=True)
class EdgeSettings:
    threshold: float = 0.27
    minimum_width_mm: float = 8
    bridge_mm: float = 2
    continuity_mm: float = 3

    @classmethod
    def parse(cls, data):
        value = overlay(cls(), data, "Strip edge settings")
        return cls(
            threshold=finite_number(value["threshold"], "Edge reflectance threshold", 0.01, 2),
            minimum_width_mm=finite_number(
                value["minimum_width_mm"], "Minimum material width", 0.01, 5000
            ),
            bridge_mm=finite_number(value["bridge_mm"], "Edge bridge", 0, 100),
            continuity_mm=finite_number(
                value["continuity_mm"], "Edge continuity tolerance", 0, 100
            ),
        )


@dataclass(frozen=True)
class SpotSettings:
    contrast: float = 0.11
    background_mm: float = 4
    minimum_area_mm2: float = 1.6
    edge_guard_mm: float = 2.5

    @classmethod
    def parse(cls, data):
        value = overlay(cls(), data, "Spot settings")
        return cls(
            contrast=finite_number(value["contrast"], "Minimum spot contrast", 0.01, 1),
            background_mm=finite_number(value["background_mm"], "Background scale", 0.1, 80),
            minimum_area_mm2=finite_number(
                value["minimum_area_mm2"], "Minimum spot area", 0.01, 5000
            ),
            edge_guard_mm=finite_number(value["edge_guard_mm"], "Edge guard", 0, 250),
        )


@dataclass(frozen=True)
class StreakSettings:
    contrast: float = 0.025
    background_mm: float = 4.0
    minimum_length_mm: float = 30.0
    minimum_coverage: float = 0.85

    @classmethod
    def parse(cls, data):
        value = overlay(cls(), data, "Streak settings")
        return cls(
            contrast=finite_number(value["contrast"], "Minimum streak contrast", 0.001, 0.6),
            background_mm=finite_number(value["background_mm"], "Streak background scale", 0.1, 40),
            minimum_length_mm=finite_number(
                value["minimum_length_mm"], "Minimum streak length", 0.1, 50_000
            ),
            minimum_coverage=finite_number(
                value["minimum_coverage"], "Minimum streak evidence", 0.25, 1
            ),
        )


@dataclass(frozen=True)
class MarkSettings:
    expected_spacing_mm: float = 60
    minimum_prominence: float = 0.06
    band_start_fraction: float = 0.72
    band_end_fraction: float = 0.9
    second_spacing_mm: float | None = None
    minimum_segment_mm: float = 240

    @classmethod
    def parse(cls, data):
        value = overlay(cls(), data, "Repeat mark settings")
        start = finite_number(value["band_start_fraction"], "Mark band start", 0.0, 1)
        end = finite_number(value["band_end_fraction"], "Mark band end", 0.0, 1)
        if start + 0.01 > end:
            raise UserInputError("The mark band must span at least 1% of the strip width.")
        expected = finite_number(
            value["expected_spacing_mm"], "Expected mark spacing", 0.1, 50_000
        )
        second = value["second_spacing_mm"]
        if second is not None:
            second = finite_number(second, "Second mark spacing", 0.1, 50_000)
            if second == expected:
                raise UserInputError(
                    "The second mark spacing must differ from the expected spacing."
                )
        segment = value["minimum_segment_mm"]
        if segment is None:
            segment = cls().minimum_segment_mm
        else:
            segment = finite_number(segment, "Minimum stable segment", 0, 100_000)
        if second is not None and segment < max(expected, second):
            raise UserInputError(
                "The minimum stable segment must cover at least one interval of the "
                "larger spacing."
            )
        return cls(
            expected_spacing_mm=expected,
            minimum_prominence=finite_number(
                value["minimum_prominence"], "Mark prominence", 0.001, 1
            ),
            band_start_fraction=start,
            band_end_fraction=end,
            second_spacing_mm=second,
            minimum_segment_mm=segment,
        )


@dataclass(frozen=True)
class TextureSettings:
    tile_size: int = 32
    overlap_fraction: float = 0.5
    minimum_wave_mm: float = 0.4
    maximum_wave_mm: float = 14

    @classmethod
    def parse(cls, data):
        value = overlay(cls(), data, "Texture settings")
        minimum = finite_number(value["minimum_wave_mm"], "Shortest wavelength", 0.05, 250)
        maximum = finite_number(value["maximum_wave_mm"], "Longest wavelength", 0.05, 500)
        if maximum <= minimum:
            raise UserInputError("The longest wavelength must exceed the shortest.")
        return cls(
            tile_size=whole_number(value["tile_size"], "Texture tile size", 8, 128),
            overlap_fraction=finite_number(value["overlap_fraction"], "Tile overlap", 0, 0.9),
            minimum_wave_mm=minimum,
            maximum_wave_mm=maximum,
        )


@dataclass(frozen=True)
class SpliceSettings:
    contrast: float = 0.08
    minimum_fraction: float = 0.65
    local_scale_mm: float = 4.0
    minimum_spacing_mm: float = 16

    @classmethod
    def parse(cls, data):
        value = overlay(cls(), data, "Splice settings")
        return cls(
            contrast=finite_number(value["contrast"], "Minimum splice contrast", 0.01, 1),
            minimum_fraction=finite_number(value["minimum_fraction"], "Width fraction", 0.2, 1),
            local_scale_mm=finite_number(value["local_scale_mm"], "Splice scale", 0.1, 75),
            minimum_spacing_mm=finite_number(
                value["minimum_spacing_mm"], "Minimum splice spacing", 0.1, 5000
            ),
        )


SETTINGS_TYPES = {
    "calibration": CalibrationSettings,
    "grid": GridSettings,
    "edges": EdgeSettings,
    "spots": SpotSettings,
    "streaks": StreakSettings,
    "marks": MarkSettings,
    "texture": TextureSettings,
    "splices": SpliceSettings,
}


@dataclass(frozen=True)
class AnalysisSettings:
    calibration: CalibrationSettings
    grid: GridSettings
    edges: EdgeSettings
    spots: SpotSettings
    streaks: StreakSettings
    marks: MarkSettings
    texture: TextureSettings
    splices: SpliceSettings

    @classmethod
    def parse(cls, data):
        data = expect_object(data, "Analysis settings")
        reject_unknown(data, SETTINGS_TYPES, "Analysis settings")
        return cls(
            **{
                section: settings_type.parse(data.get(section, {}))
                for section, settings_type in SETTINGS_TYPES.items()
            }
        )

    @classmethod
    def defaults(cls):
        return cls.parse({})

    def as_dict(self):
        return asdict(self)

    def with_section(self, section, values):
        reject_unknown({section: None}, SETTINGS_TYPES, "Analysis settings")
        combined = self.as_dict()
        combined[section].update(expect_object(values, section))
        return self.parse(combined)
