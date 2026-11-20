import io
import json
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from rollscope.analysis import AnalysisContext
from rollscope.demo import PROFILES, build_demo, encoder_program, references
from rollscope.domain import Capture, CaptureMetadata, Geometry, StationWindow, UserInputError
from rollscope.grain import (
    compare_grain,
    grain_axes,
    presentation,
    window_tiles,
)
from rollscope.http import create_app
from rollscope.settings import AnalysisSettings, GrainSettings


class RollingWindow:
    def __init__(self, identifier, label, region):
        self.id = identifier
        self.label = label
        self.region = region


def source(role, context, region, label):
    from rollscope.grain import GrainSource

    return GrainSource(role, "capture", context, RollingWindow(label, label, region))


def textured_capture(angle_deg=0.0, scale=1.0, brightness_swing=0.0, seed=999):
    """Cold-rolled demo geometry with a controlled grain angle and coarseness.

    Zero degrees reproduces the demo’s own 2.6° measured stripe direction;
    each additional degree rotates the grain by that much.
    """
    profile = PROFILES["cold-rolled"]
    rng = np.random.default_rng(seed)
    geometry = Geometry(
        profile.encoder_pitch, profile.transverse_pitch, profile.station_origin, 0.0
    )
    ticks, seconds = encoder_program(profile, rng)
    dark_level, sensitivity, dark, white = references(profile, rng)
    along = (geometry.station(ticks) - geometry.station(ticks)[0])[:, None]
    across = np.arange(profile.columns)[None, :] * profile.transverse_pitch
    theta = np.radians(angle_deg)
    u = across * np.cos(theta) + along * np.sin(theta)
    v = -across * np.sin(theta) + along * np.cos(theta)
    texture = (
        0.022 * np.sin(2 * np.pi * (u / (1.24 * scale) + v / 10.7))
        + 0.012 * np.cos(2 * np.pi * (u / (3.87 * scale) - v / 12.9))
        + 0.0051 * np.sin(2 * np.pi * (u / (0.88 * scale) + v / 2.95))
    )
    swing = brightness_swing * np.sin(2 * np.pi * along / 180.0) * np.cos(across / across.max())
    reflectance = 0.574 + texture + swing
    backdrop = 0.084 + 0.017 * np.cos(np.arange(profile.columns) / 31)
    support = np.clip(0.6 + np.minimum(across - 31.5, 288 - across), 0, 1)
    image = (
        backdrop * (1 - support)
        + reflectance * support
        + rng.normal(0, 0.0029, size=(profile.rows, profile.columns))
    )
    raw = np.rint(dark_level + sensitivity * image).clip(0, 65535).astype(np.uint16)
    metadata = CaptureMetadata("Grain stand-in", "Cold-rolled steel", "", True, geometry)
    return Capture.build(raw, ticks, seconds, dark, white, metadata)


def compare(context_a, context_b, context_target, settings, regions):
    (region_a, region_b, region_target) = regions
    return compare_grain(
        settings,
        [
            source("reference_a", context_a, region_a, "A"),
            source("reference_b", context_b, region_b, "B"),
            source("target", context_target, region_target, "T"),
        ],
    )


class GrainTiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = build_demo("cold-rolled")
        cls.settings = AnalysisSettings.defaults()
        cls.context = AnalysisContext(cls.capture, cls.settings)

    def test_tile_is_physical_size_on_each_axis(self):
        window = window_tiles(
            self.context,
            self.settings.grain,
            StationWindow(1400, 1600, 8, 80),
            "w",
            "W",
        )
        grid = self.context.grid
        self.assertEqual(window.tile_rows * grid.row_pitch_mm, window.region["tile_height_mm"])
        self.assertEqual(window.tile_columns * grid.column_pitch_mm, window.region["tile_width_mm"])
        self.assertAlmostEqual(window.region["tile_height_mm"], self.settings.grain.tile_size_mm, delta=0.8)
        self.assertAlmostEqual(window.region["tile_width_mm"], self.settings.grain.tile_size_mm, delta=0.8)
        for tile in window.tiles[:20]:
            # Encoder ticks wobble around the nominal pitch, so a tile's observed
            # station span may differ slightly from the pitch used by its FFT.
            self.assertAlmostEqual(
                tile.height_mm,
                (tile.row_stop - 1 - tile.row_start) * grid.row_pitch_mm,
                delta=4 * grid.row_pitch_mm,
            )
            self.assertAlmostEqual(
                tile.width_mm, (tile.column_stop - 1 - tile.column_start) * grid.column_pitch_mm,
                delta=2 * grid.column_pitch_mm,
            )

    def test_frequency_axes_use_millimetre_pitches(self):
        axes = grain_axes(16, 40, 0.72, 0.28, 0.4, 8.0)
        recovered = 1 / np.array([0.72 * 3, 0.28 * 5])
        self.assertTrue(np.all(np.isfinite(axes.wavelength_mm)))
        self.assertTrue(np.all(axes.wavelength_mm >= 0.4 - 1e-9))
        self.assertTrue(np.all(axes.wavelength_mm <= 8.0 + 1e-9))
        self.assertTrue(np.any(np.isclose(axes.wavelength_mm, recovered[0], atol=0.3)))
        self.assertTrue(np.any(np.isclose(axes.wavelength_mm, recovered[1], atol=0.3)))

    def test_unsupported_tiles_are_unknown_not_filled(self):
        window = window_tiles(
            self.context,
            self.settings.grain,
            StationWindow(1400, 1600, 8, 80),
            "w",
            "W",
        )
        unsupported = [tile for tile in window.tiles if tile.state.value == "unsupported_sampling"]
        self.assertTrue(unsupported)
        for tile in unsupported:
            self.assertIsNone(tile.direction_deg)
            self.assertIsNone(tile.coarseness_mm)
            self.assertIsNone(tile.relative_amplitude)
            self.assertFalse(self.context.grid.valid[tile.row_start : tile.row_stop, tile.column_start : tile.column_stop].all())


class GrainComparisonBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = AnalysisSettings.defaults()
        cls.reference = AnalysisContext(build_demo("cold-rolled"), cls.settings)

    def regions(self):
        return (
            StationWindow(1250, 1450, 5, 82),
            StationWindow(1550, 1760, 6, 84),
            StationWindow(1600, 1800, 6, 84),
        )

    def test_reference_window_matches_itself(self):
        comparison = compare(
            self.reference, self.reference, self.reference, self.settings.grain, self.regions()
        )
        summary = comparison.summary()
        self.assertGreater(summary["both_references_fraction"], 0.93)
        self.assertLess(summary["outside_normal_fraction"], 0.07)
        self.assertEqual(summary["reference_a_status"], "usable")

    def test_lighting_swing_does_not_change_the_classification(self):
        baseline = compare(
            self.reference, self.reference, self.reference, self.settings.grain, self.regions()
        ).summary()
        self.assertGreater(baseline["both_references_fraction"], 0.93)
        flat = AnalysisContext(textured_capture(0.0, 1.0, 0.0), self.settings)
        reference = compare(
            self.reference, self.reference, flat, self.settings.grain, self.regions()
        ).summary()
        self.assertGreater(reference["both_references_fraction"], 0.9)
        for swing in (0.05, 0.12, 0.2):
            shifted = AnalysisContext(textured_capture(0.0, 1.0, swing), self.settings)
            summary = compare(
                self.reference, self.reference, shifted, self.settings.grain, self.regions()
            ).summary()
            with self.subTest(swing=swing):
                self.assertAlmostEqual(
                    summary["both_references_fraction"], reference["both_references_fraction"], delta=0.01
                )
                self.assertEqual(summary["neither_reference_tiles"], reference["neither_reference_tiles"])

    def test_direction_shift_is_separated_per_reference(self):
        first = AnalysisContext(textured_capture(-1.5), self.settings)
        second = AnalysisContext(textured_capture(2.5), self.settings)
        for angle, expected in (
            (-1.5, "only_reference_a"),
            (2.5, "only_reference_b"),
            (6.5, "neither_reference"),
        ):
            target = AnalysisContext(textured_capture(angle), self.settings)
            comparison = compare(first, second, target, self.settings.grain, self.regions())
            counts = comparison.matrix()["target"]
            with self.subTest(angle=angle):
                self.assertEqual(counts[expected], max(counts[c] for c in counts if c != "unknown"))

    def test_coarseness_shift_is_detected(self):
        target = AnalysisContext(textured_capture(0.0, scale=1.4), self.settings)
        comparison = compare(
            self.reference, self.reference, target, self.settings.grain, self.regions()
        )
        counts = comparison.matrix()["target"]
        self.assertGreater(counts["neither_reference"], 0.8 * sum(
            value for key, value in counts.items() if key != "unknown"
        ))

    def test_insufficient_reference_cannot_classify(self):
        settings = GrainSettings(minimum_reference_tiles=5000)
        comparison = compare(
            self.reference, self.reference, self.reference, settings, self.regions()
        )
        self.assertEqual(comparison.envelope_a.status, "insufficient")
        self.assertFalse(comparison.envelope_a.usable)
        self.assertEqual(
            {row["classification"] for row in comparison.rows}, {"unknown"}
        )

    def test_evidence_confirms_tile_origin_in_physical_units(self):
        comparison = compare(
            self.reference, self.reference, self.reference, self.settings.grain, self.regions()
        )
        tile = comparison.target.tiles[0]
        evidence = presentation(comparison)["tables"][0]["rows"][0]
        self.assertAlmostEqual(evidence["station_mm"], tile.station_mm)
        self.assertAlmostEqual(
            tile.station_to_mm - tile.station_from_mm, self.settings.grain.tile_size_mm, delta=1.5
        )
        self.assertLessEqual(tile.station_from_mm, evidence["station_mm"])
        self.assertGreaterEqual(tile.station_to_mm, evidence["station_mm"])


class GrainApi(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.client = create_app(self.temporary.name).test_client()
        self.ref = self.client.get("/api/captures").json["captures"][0]["id"]
        self.api = f"/api/captures/{self.ref}"

    def make_window(self, revision, label, region):
        response = self.client.post(
            f"{self.api}/windows",
            json={
                "revision": revision,
                "label": label,
                "note": "",
                "status": "open",
                "region": region,
            },
        )
        self.assertEqual(response.status_code, 201, response.json)
        return response.json["window"]["id"], response.json["state"]["revision"]

    def configure(self):
        revision = 0
        first, revision = self.make_window(
            revision,
            "Signed off A",
            {"station_start_mm": 1250, "station_end_mm": 1450, "transverse_start_mm": 5, "transverse_end_mm": 82},
        )
        second, revision = self.make_window(
            revision,
            "Signed off B",
            {"station_start_mm": 1550, "station_end_mm": 1760, "transverse_start_mm": 6, "transverse_end_mm": 84},
        )
        target, revision = self.make_window(
            revision,
            "New roll",
            {"station_start_mm": 1820, "station_end_mm": 2030, "transverse_start_mm": 6, "transverse_end_mm": 84},
        )
        payload = {
            "revision": revision,
            "reference_a": {"capture": self.ref, "window": first},
            "reference_b": {"capture": self.ref, "window": second},
            "target": {"capture": self.ref, "window": target},
        }
        stored = self.client.put(f"{self.api}/grain", json=payload)
        self.assertEqual(stored.status_code, 200, stored.json)
        return first, second, target, stored.json["state"]["revision"]

    def test_unconfigured_state_lists_windows(self):
        response = self.client.get(f"{self.api}/grain")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["configured"])
        self.assertEqual(response.json["catalog"], [])

    def test_configured_comparison_and_tile_evidence(self):
        number_a, number_b, target, _ = self.configure()
        result = self.client.get(f"{self.api}/grain")
        self.assertEqual(result.status_code, 200, result.json)
        body = result.json
        self.assertTrue(body["configured"])
        self.assertEqual(len(body["catalog"]), 3)
        self.assertEqual(body["summary"]["target_window"], "New roll")
        self.assertEqual(
            {row["classification"] for row in body["tables"][0]["rows"]}
            <= {"both_references", "only_reference_a", "only_reference_b", "neither_reference", "unknown"},
            True,
        )
        rows = body["tables"][0]["rows"]
        measured = [row for row in rows if row["classification"] != "unknown"]
        self.assertTrue(measured)
        tile = self.client.get(f"{self.api}/grain/tiles/target/{measured[0]['tile']}")
        self.assertEqual(tile.status_code, 200, tile.json)
        detail = tile.json
        self.assertEqual(detail["window"], "target")
        self.assertTrue(detail["source_lines"])
        self.assertAlmostEqual(detail["tile"]["station_mm"], measured[0]["station_mm"])
        self.assertLessEqual(
            detail["tile"]["transverse_from_mm"], detail["tile"]["transverse_mm"]
        )
        self.assertGreaterEqual(
            detail["tile"]["transverse_to_mm"], detail["tile"]["transverse_mm"]
        )

    def test_classification_preview_is_coloured(self):
        self.configure()
        image = self.client.get(f"{self.api}/preview.png?layer=grain")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.data[:8], b"\x89PNG\r\n\x1a\n")

    def test_export_contains_tiles_bands_and_source_samples(self):
        self.configure()
        export = self.client.get(f"{self.api}/grain/export")
        self.assertEqual(export.status_code, 200)
        with ZipFile(io.BytesIO(export.data)) as archive:
            names = set(archive.namelist())
            for member in (
                "inspection.json",
                "target-tiles.csv",
                "reference-a-tiles.csv",
                "reference-b-tiles.csv",
                "envelopes.csv",
                "wavelength-bands.csv",
                "source-target.npz",
                "classification.png",
            ):
                self.assertIn(member, names)
            with np.load(io.BytesIO(archive.read("source-target.npz"))) as data:
                lines = data["capture_lines"]
                columns = data["capture_columns"]
                source = np.load(io.BytesIO(self.client.get(f"{self.api}/source").data))
                np.testing.assert_array_equal(
                    data["raw"], source["raw"][np.ix_(lines, columns)]
                )

    def test_signed_off_windows_can_live_on_another_capture(self):
        first, second, _, _ = self.configure()
        roll = self.client.post("/api/captures/demo", json={"profile": "coated"}).json["id"]
        created = self.client.post(
            f"/api/captures/{roll}/windows",
            json={
                "revision": 0,
                "label": "New roll window",
                "note": "",
                "status": "open",
                "region": {
                    "station_start_mm": 400,
                    "station_end_mm": 600,
                    "transverse_start_mm": 10,
                    "transverse_end_mm": 60,
                },
            },
        )
        target = created.json["window"]["id"]
        revision = created.json["state"]["revision"]
        stored = self.client.put(
            f"/api/captures/{roll}/grain",
            json={
                "revision": revision,
                "reference_a": {"capture": self.ref, "window": first},
                "reference_b": {"capture": self.ref, "window": second},
                "target": {"capture": roll, "window": target},
            },
        )
        self.assertEqual(stored.status_code, 200, stored.json)
        result = self.client.get(f"/api/captures/{roll}/grain").json
        self.assertTrue(result["configured"])
        self.assertEqual(result["target_capture"], roll)
        self.assertGreater(result["summary"]["measured_tiles"], 0)
        image = self.client.get(f"/api/captures/{roll}/preview.png?layer=grain")
        self.assertEqual(image.status_code, 200)

    def test_project_backup_restores_a_self_contained_link(self):
        self.configure()
        archive = self.client.get(f"{self.api}/project").data
        restored = self.client.post(
            "/api/captures/restore", data={"project": (io.BytesIO(archive), "backup.zip")}
        )
        self.assertEqual(restored.status_code, 201, restored.json)
        copy = restored.json["id"]
        result = self.client.get(f"/api/captures/{copy}/grain")
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json["configured"])
        self.assertEqual(result.json["target_capture"], copy)
        for role in ("reference_a", "reference_b", "target"):
            self.assertEqual(result.json["selection"][role]["capture"], copy)

    def test_duplicate_references_are_rejected(self):
        first, _, target, revision = self.configure()
        response = self.client.put(
            f"{self.api}/grain",
            json={
                "revision": revision,
                "reference_a": {"capture": self.ref, "window": first},
                "reference_b": {"capture": self.ref, "window": first},
                "target": {"capture": self.ref, "window": target},
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_window_is_reported(self):
        first, second, _, revision = self.configure()
        response = self.client.put(
            f"{self.api}/grain",
            json={
                "revision": revision,
                "reference_a": {"capture": self.ref, "window": first},
                "reference_b": {"capture": self.ref, "window": second},
                "target": {"capture": self.ref, "window": "missing"},
            },
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
