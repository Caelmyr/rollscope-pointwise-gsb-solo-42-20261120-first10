import io
import json
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from rollscope.analysis import SECTION_TITLES, AnalysisContext
from rollscope.capture_codec import capture_bytes, capture_hash, read_capture
from rollscope.demo import build_demo
from rollscope.domain import Capture, CaptureMetadata, Geometry, StationWindow
from rollscope.grid import Support
from rollscope.http import create_app
from rollscope.lineage import station_lineage
from rollscope.ruler import Ruler, evaluate_ruler
from rollscope.settings import AnalysisSettings
from rollscope.smear import SmearState, blur_operator, solve


def variable_speed_capture():
    """A capture that is fast in the middle, with a repeated tick and a pause.

    A fine dark line pattern is convolved with each line's exposure displacement
    in material coordinates, so the same mark is sharp where the roll moved slowly
    and spread out where it moved fast.
    """
    rng = np.random.default_rng(11)
    rows, columns = 900, 60
    tick_mm, column_mm = 0.05, 0.25
    increments = np.full(rows, 2, dtype=np.int64)
    increments[0] = 0
    increments[200:380] = 7
    increments[380:400] = 2
    increments[650] = 0
    ticks = np.cumsum(increments)
    times = np.full(rows, 0.004)
    times[0] = 0
    times[650] = 0.15
    seconds = np.cumsum(times)
    stations = 1200 + ticks * tick_mm
    material = np.arange(stations[0] - 0.5, stations[-1] + 0.5, 0.002)
    truth = np.full_like(material, 0.55)
    for center in np.arange(material[0] + 4, material[-1] - 3, 5.0):
        truth -= 0.24 * np.exp(-0.5 * ((material - center) / 0.10) ** 2)
    truth += 0.005 * np.sin(material / 3.1)
    width = np.where(increments > 0, increments * tick_mm, 0.1)
    width[0] = width[1]
    along = np.array(
        [
            truth[(material >= center - width[i] / 2) & (material <= center + width[i] / 2)].mean()
            for i, center in enumerate(stations)
        ]
    )
    across = np.arange(columns) * column_mm
    reflectance = along[:, None] + 0.004 * np.sin(across / 1.7)[None, :]
    raw = np.rint(820 + 39600 * reflectance).clip(0, 65535).astype(np.uint16)
    dark = np.rint(820 + rng.normal(0, 6, (5, columns))).clip(0, 65535).astype(np.uint16)
    white = np.rint(820 + 39600 + rng.normal(0, 18, (6, columns))).clip(0, 65535).astype(np.uint16)
    metadata = CaptureMetadata(
        "Variable speed", "Synthetic", "Speed change mid-capture", True,
        Geometry(tick_mm, column_mm, 1200.0, 0.0),
    )
    return Capture.build(raw, ticks, seconds, dark, white, metadata), material, truth


class MotionSmear(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture, cls.material, cls.truth = variable_speed_capture()
        cls.context = AnalysisContext(cls.capture, AnalysisSettings.defaults())

    def window(self):
        return StationWindow.full(self.capture)

    def test_exposure_changes_with_speed(self):
        profile = self.context.transport
        slow_lines = np.arange(0, 200)
        fast_lines = np.arange(200, 380)
        distance = profile.exposure_distance_mm()
        slow_distance = np.nanmedian(distance[slow_lines])
        fast_distance = np.nanmedian(distance[fast_lines])
        self.assertAlmostEqual(slow_distance, 0.10, places=2)
        self.assertGreater(fast_distance, 2.5 * slow_distance)

    def test_interruptions_keep_an_unknown_exposure(self):
        profile = self.context.transport
        repeated = 650
        self.assertFalse(profile.known[repeated])
        self.assertFalse(np.isfinite(profile.exposure_distance_mm()[repeated]))
        report = self.context.smear_window(self.window())
        self.assertGreater(report.state_counts()["unknown_exposure"], 0)
        self.assertTrue(
            any(row.state == SmearState.UNKNOWN_EXPOSURE for row in report.row_ranges)
        )

    def test_restoration_is_per_row_not_one_factor(self):
        report = self.context.smear_window(self.window())
        grid = self.context.grid
        slow_rows = np.flatnonzero(grid.stations <= self.capture.stations[150])
        fast_rows = np.flatnonzero(
            (grid.stations >= self.capture.stations[220])
            & (grid.stations <= self.capture.stations[360])
        )
        slow_blur = np.median(report.blur_rows[slow_rows])
        fast_blur = np.median(report.blur_rows[fast_rows])
        self.assertAlmostEqual(slow_blur, 1.0, delta=0.35)
        self.assertGreater(fast_blur, 2.5)
        self.assertGreater(len(np.unique(np.round(report.blur_rows, 2))), 5)

    def test_fast_marks_gain_depth_and_slow_marks_are_unchanged(self):
        report = self.context.smear_window(self.window())
        grid = self.context.grid
        truth = np.interp(grid.stations, self.material, self.truth)
        fast = np.flatnonzero(
            (grid.stations >= self.capture.stations[215])
            & (grid.stations <= self.capture.stations[365])
        )
        slow = np.flatnonzero(
            (grid.stations >= self.capture.stations[20])
            & (grid.stations <= self.capture.stations[150])
        )
        recorded_fast = report.original[fast, 0]
        restored_fast = report.restored[fast, 0]
        self.assertTrue(np.all(np.isfinite(restored_fast)))
        self.assertLess(restored_fast.min(), recorded_fast.min() - 0.02)
        restored_error = np.sqrt(np.mean((restored_fast - truth[fast]) ** 2))
        recorded_error = np.sqrt(np.mean((recorded_fast - truth[fast]) ** 2))
        self.assertLess(restored_error, recorded_error)
        recorded_slow = report.original[slow, 0]
        restored_slow = report.restored[slow, 0]
        self.assertTrue(np.allclose(recorded_slow, restored_slow, atol=0.01))

    def test_identity_kernel_and_unrestored_rows_are_not_invented(self):
        values = np.tile(np.linspace(0.2, 0.8, 40)[:, None], (1, 5))
        operator = blur_operator(
            type(
                "Kernels",
                (),
                {
                    "starts": np.arange(41),
                    "offsets": np.zeros(40),
                    "widths": np.ones(40),
                    "weights": np.ones(40),
                },
            )(),
            slice(0, 40),
        )
        outcome, _ = solve(operator, values, 16)
        self.assertTrue(np.allclose(outcome, values))
        report = self.context.smear_window(self.window())
        blocked = report.state != SmearState.RESTORED
        self.assertTrue(np.isnan(report.restored[blocked]).all())
        self.assertTrue(np.isfinite(report.original[blocked]).any())

    def test_window_probe_returns_recorded_source_lines(self):
        from rollscope.probe import point_measurement
        from rollscope.raster import Layer, RasterSpec

        spec = RasterSpec(Layer.SMEAR, window=self.window())
        result = point_measurement(self.context, spec, 0.5, 0.5)
        self.assertEqual(result["layer"], "smear")
        self.assertIn(result["coverage"], {state.name.lower() for state in SmearState})
        self.assertGreater(result["lineage"]["line_count"], 0)
        for line in result["lineage"]["segments"][0]["rows"]:
            self.assertIn("source_line", line)
            self.assertIn("sensor_code", line)
            self.assertIn("exposure_ms", line)
        recorded = point_measurement(self.context, RasterSpec(
            Layer.SMEAR, window=self.window(), recorded=True
        ), 0.5, 0.5)
        self.assertEqual(recorded["displayed"], "recorded")
        self.assertEqual(recorded["reflectance"], result["reflectance"])


    def test_motion_smear_window_route_requires_a_window(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        app = create_app(temporary.name)
        client = app.test_client()
        ref = client.get("/api/captures").json["captures"][0]["id"]
        api = f"/api/captures/{ref}"
        missing = client.get(f"{api}/preview.png?layer=smear&revision=0")
        self.assertEqual(missing.status_code, 400)
        added = client.post(f"{api}/windows", json={
            "revision": 0,
            "label": "Transport change",
            "note": "",
            "status": "watch",
            "region": {
                "station_start_mm": 1450.0,
                "station_end_mm": 1540.0,
                "transverse_start_mm": 8.0,
                "transverse_end_mm": 80.0,
            },
        })
        window = added.json["window"]["id"]
        revision = added.json["state"]["revision"]
        query = f"layer=smear&window={window}&revision={revision}"
        restored = client.get(f"{api}/preview.png?{query}")
        recorded = client.get(f"{api}/preview.png?{query}&recorded=1")
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(recorded.status_code, 200)
        self.assertEqual(
            restored.headers["X-Preview-Shape"], recorded.headers["X-Preview-Shape"]
        )
        report = client.get(f"{api}/smear?revision={revision}").json["windows"]
        self.assertEqual(len(report), 1)
        self.assertEqual(
            report[0]["states"]["restored"], report[0]["summary"]["restored_rows"]
        )
        probe = client.get(f"{api}/probe?{query}&x=0.5&y=0.5").json
        self.assertEqual(probe["layer"], "smear")
        self.assertIn("recorded_reflectance", probe)


class Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = build_demo("cold-rolled")
        cls.context = AnalysisContext(cls.capture, AnalysisSettings.defaults())

    def test_bundle_reproduces_source_identity_and_codes(self):
        original = self.capture
        restored = read_capture(capture_bytes(original))
        self.assertEqual(capture_hash(original), capture_hash(restored))
        self.assertEqual(restored.raw.dtype, np.dtype("<u2"))
        np.testing.assert_array_equal(restored.raw, original.raw)
        np.testing.assert_array_equal(restored.ticks, original.ticks)

    def test_invalid_optics_do_not_become_station_samples(self):
        context = self.context
        self.assertEqual(context.calibrated.reference.summary()["rejected_columns"], 3)
        self.assertTrue(np.all(~context.grid.valid[:, 22]))
        self.assertTrue(np.isnan(context.grid.values[:, 22]).all())
        self.assertEqual(len(context.coverage.gaps), 2)

    def test_repeat_and_blend_provenance_reconstruct_pixels(self):
        context = self.context
        grid = context.grid
        column = 100
        native = int(np.flatnonzero((grid.plan.support == Support.NATIVE) & grid.valid[:, column])[5])
        blended = int(np.flatnonzero((grid.plan.support == Support.BLEND) & grid.valid[:, column])[5])
        gap = int(np.flatnonzero(grid.plan.support == Support.GAP)[0])
        for index in (native, blended, gap):
            records = station_lineage(context, index, column)
            weight = sum(row["effective_weight"] for row in records)
            if index == gap:
                self.assertEqual(weight, 0)
                self.assertFalse(grid.valid[index, column])
            else:
                self.assertAlmostEqual(weight, 1)
                total = sum(
                    row["effective_weight"] * row["reflectance"]
                    for row in records if row["effective_weight"] > 0
                )
                self.assertAlmostEqual(grid.values[index, column], total, places=6)

    def test_ruler_conserves_length_and_keeps_unknown_interval(self):
        context = self.context
        result = evaluate_ruler(
            context,
            Ruler.parse({
                "start": {"station_mm": 1450.0, "transverse_mm": 30.0},
                "end": {"station_mm": 1540.0, "transverse_mm": 35.0},
            }, context.grid),
        ).summary()
        self.assertAlmostEqual(result["length_mm"], sum(result["distance_by_support_mm"].values()))
        self.assertGreater(result["distance_by_support_mm"]["gap"], 0)


class WorkspaceLifecycle(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.app = create_app(self.temporary.name)
        self.client = self.app.test_client()
        self.ref = self.client.get("/api/captures").json["captures"][0]["id"]
        self.api = f"/api/captures/{self.ref}"

    def test_default_reports_and_images(self):
        for section in SECTION_TITLES:
            with self.subTest(section=section):
                response = self.client.get(f"{self.api}/analysis/{section}")
                self.assertEqual(response.status_code, 200)
                self.assertIsInstance(response.json["tables"], list)
        image = self.client.get(f"{self.api}/preview.png?layer=coverage")
        self.assertEqual(image.headers["X-Preview-Shape"], "320x1156")
        self.assertEqual(image.data[:8], b"\x89PNG\r\n\x1a\n")

    def test_revision_conflict_leaves_previous_notes(self):
        accepted = self.client.put(f"{self.api}/notes", json={"revision": 0, "notes": "Keep original"})
        self.assertEqual(accepted.status_code, 200)
        conflict = self.client.put(f"{self.api}/notes", json={"revision": 0, "notes": "Desynchronized"})
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(self.client.get(self.api).json["state"]["notes"], "Keep original")

    def test_window_evidence_and_project_restore(self):
        added = self.client.post(f"{self.api}/windows", json={
            "revision": 0,
            "label": "First transport interruption",
            "note": "Synthetic control",
            "status": "watch",
            "region": {
                "station_start_mm": 1450.0, "station_end_mm": 1540.0,
                "transverse_start_mm": 8.0, "transverse_end_mm": 80.0,
            },
        })
        self.assertEqual(added.status_code, 201, added.json)
        window = added.json["window"]["id"]
        result = self.client.get(f"{self.api}/windows/{window}")
        self.assertEqual(result.status_code, 200, result.json)
        self.assertEqual(result.json["summary"]["gap_count"], 1)
        self.assertTrue(any(band["status"] == "encoder_gap"
                            for band in result.json["profiles"]["station_bands"]))
        exported = self.client.get(f"{self.api}/windows/{window}/export")
        self.assertEqual(exported.status_code, 200)
        with ZipFile(io.BytesIO(exported.data)) as archive:
            self.assertIn("review-sheet.png", archive.namelist())
            with np.load(io.BytesIO(archive.read("station-cells.npz"))) as data:
                unknown = data["support"] == Support.GAP
                self.assertTrue(np.any(unknown))
                self.assertFalse(np.any(data["valid"][unknown]))
                self.assertTrue(np.isnan(data["reflectance"][unknown]).all())

        snap = self.client.post(f"{self.api}/snapshots", json={
            "revision": 1, "section": "flat", "label": "Flat-field check",
        })
        self.assertEqual(snap.status_code, 201)
        stored_archive = self.client.get(f"{self.api}/project").data
        restored = self.client.post("/api/captures/restore", data={
            "project": (io.BytesIO(stored_archive), "roundtrip.zip"),
        })
        self.assertEqual(restored.status_code, 201, restored.json)
        other = self.client.get(f"/api/captures/{restored.json['id']}").json
        current = self.client.get(self.api).json
        self.assertEqual(other["sha256"], current["sha256"])
        self.assertNotEqual(other["id"], current["id"])
        self.assertEqual(other["state"], current["state"])
        self.assertEqual(other["snapshots"][0]["label"], current["snapshots"][0]["label"])
        self.assertNotEqual(other["snapshots"][0]["id"], current["snapshots"][0]["id"])

    def test_bad_upload_returns_input_error(self):
        rejected = self.client.post("/api/captures/import", data={
            "capture": (io.BytesIO(b"not a NumPy archive"), "roll.npz"),
        })
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(len(self.client.get("/api/captures").json["captures"]), 1)


if __name__ == "__main__":
    unittest.main()
