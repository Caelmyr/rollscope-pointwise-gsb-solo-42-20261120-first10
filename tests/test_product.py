import io
import json
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from rollscope.analysis import SECTION_TITLES, AnalysisContext
from rollscope.capture_codec import capture_bytes, capture_hash, read_capture
from rollscope.demo import build_demo, ruler_physical_map, PROFILES
from rollscope.grid import Support
from rollscope.http import create_app
from rollscope.lineage import station_lineage
from rollscope.ruler import Ruler, evaluate_ruler
from rollscope.settings import AnalysisSettings
from rollscope.transverse import (
    TransverseCalibration,
    build_transverse_calibration,
    detect_ruler_lines,
)


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


class TransverseCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = build_demo("ruler")
        cls.lines, _ = detect_ruler_lines(cls.capture.raw)
        physical = ruler_physical_map(PROFILES["ruler"])
        spacing, margin = 14.0, 10.0
        first = float(np.ceil((physical[0] + margin) / spacing) * spacing)
        last = float(np.floor((physical[-1] - margin) / spacing) * spacing)
        cls.positions = list(np.arange(first, last + 0.01, spacing))
        cls.physical = physical

    def test_detects_tilted_lines_and_flags_the_stain(self):
        self.assertEqual(len(self.lines), len(self.positions))
        stained = [line.line_id for line in self.lines if line.stained]
        self.assertEqual(stained, [3])
        slopes = [line.slope for line in self.lines]
        self.assertTrue(any(abs(slope) > 0.005 for slope in slopes))
        self.assertTrue(all(line.coverage > 0.8 for line in self.lines))

    def test_robust_fit_rejects_local_deformation_without_bending_neighbours(self):
        calibration = build_transverse_calibration(
            self.capture.raw,
            self.lines,
            self.positions,
            None,
            {"name": "ruler", "capture_id": "x", "sha256": "y"},
        )
        outliers = [anchor["line_id"] for anchor in calibration.anchors if anchor["robust_outlier"]]
        self.assertEqual(outliers, [5])
        self.assertEqual(len(calibration.anchor_columns), len(self.positions) - 1)
        reliable = np.flatnonzero(calibration.calibrated)
        error = np.abs(calibration.transverse[reliable] - self.physical[reliable])
        self.assertLess(float(error.max()), 0.5)
        # Piecewise-linear anchors keep neighbouring widths anchored, not pulled to the bad line.
        self.assertTrue(np.all(np.diff(calibration.transverse[reliable]) > 0))

    def test_reliable_span_ends_at_outer_anchors_and_edges_are_uncalibrated(self):
        calibration = build_transverse_calibration(
            self.capture.raw,
            self.lines,
            self.positions,
            None,
            {"name": "ruler", "capture_id": "x", "sha256": "y"},
        )
        self.assertFalse(calibration.calibrated[0])
        self.assertFalse(calibration.calibrated[-1])
        self.assertGreater(calibration.first_column, 0)
        self.assertLess(calibration.last_column, calibration.columns - 1)
        self.assertFalse(calibration.is_calibrated_mm(calibration.transverse[0]))
        self.assertTrue(calibration.is_calibrated_mm(calibration.transverse[200]))
        # The inverse mapping recovers the original detector column even outside the span.
        for column in (0, 50, 200, calibration.columns - 1):
            self.assertAlmostEqual(
                calibration.column_at(calibration.transverse[column]), column, places=6
            )

    def test_serialization_round_trip_and_manual_exclusion(self):
        calibration = build_transverse_calibration(
            self.capture.raw,
            self.lines,
            self.positions,
            None,
            {"name": "ruler", "capture_id": "x", "sha256": "y"},
        )
        restored = TransverseCalibration.parse(calibration.as_dict())
        np.testing.assert_allclose(restored.transverse, calibration.transverse, atol=1e-5)
        np.testing.assert_array_equal(restored.calibrated, calibration.calibrated)
        # Manually dropping the stained line still leaves enough anchors to fit.
        kept = [line.line_id for line in self.lines if line.line_id != 3]
        manual = build_transverse_calibration(
            self.capture.raw,
            self.lines,
            self.positions,
            kept,
            {"name": "ruler", "capture_id": "x", "sha256": "y"},
        )
        self.assertEqual(len(manual.anchor_columns), len(kept) - 1)

    def test_rejects_too_few_or_misordered_lines(self):
        with self.assertRaises(Exception):
            build_transverse_calibration(
                self.capture.raw,
                self.lines,
                self.positions,
                [self.lines[0].line_id],
                {"name": "ruler", "capture_id": "x", "sha256": "y"},
            )
        bad_positions = list(self.positions)
        bad_positions[0], bad_positions[1] = bad_positions[1], bad_positions[0]
        with self.assertRaises(Exception):
            build_transverse_calibration(
                self.capture.raw,
                self.lines,
                bad_positions,
                None,
                {"name": "ruler", "capture_id": "x", "sha256": "y"},
            )


class TransverseWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.app = create_app(self.temporary.name, seed=False)
        self.client = self.app.test_client()
        record = self.app.extensions["rollscope"].make_demo("ruler")
        self.ref = record.id
        self.api = f"/api/captures/{self.ref}"
        physical = ruler_physical_map(PROFILES["ruler"])
        spacing, margin = 14.0, 10.0
        first = float(np.ceil((physical[0] + margin) / spacing) * spacing)
        last = float(np.floor((physical[-1] - margin) / spacing) * spacing)
        self.positions = list(np.arange(first, last + 0.01, spacing))

    def test_detect_build_apply_clear_and_restore(self):
        detected = self.client.get(f"{self.api}/transverse/lines")
        self.assertEqual(detected.status_code, 200)
        self.assertEqual(detected.json["line_count"], len(self.positions))
        built = self.client.post(
            f"{self.api}/transverse",
            json={"revision": 0, "positions_mm": self.positions, "accepted_lines": None},
        )
        self.assertEqual(built.status_code, 201, built.json)
        self.assertLess(built.json["calibration"]["rms_error_mm"], 0.5)

        edges = self.client.get(f"{self.api}/analysis/edges?revision=1")
        self.assertEqual(edges.status_code, 200)
        self.assertGreater(edges.json["summary"]["uncalibrated_rows"], 0)
        self.assertIn("uncalibrated", edges.json["summary"]["rows"])

        preview = self.client.get(f"{self.api}/preview.png?layer=station&revision=1")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.data[:8], b"\x89PNG\r\n\x1a\n")

        exported = self.client.get(f"{self.api}/transverse/export")
        self.assertEqual(exported.status_code, 200)
        payload = exported.json
        cleared = self.client.delete(f"{self.api}/transverse", json={"revision": 1})
        self.assertEqual(cleared.status_code, 200)
        self.assertIsNone(cleared.json["state"]["transverse_calibration"])
        reimported = self.client.put(
            f"{self.api}/transverse", json={"revision": 2, "calibration": payload}
        )
        self.assertEqual(reimported.status_code, 200)
        self.assertEqual(
            reimported.json["calibration"]["calibration_id"], payload["calibration_id"]
        )

    def test_wrong_column_count_calibration_is_rejected(self):
        detected = self.client.get(f"{self.api}/transverse/lines").json
        built = self.client.post(
            f"{self.api}/transverse",
            json={"revision": 0, "positions_mm": self.positions, "accepted_lines": None},
        ).json
        payload = built["state"]["transverse_calibration"]
        payload["columns"] = payload["columns"] - 1
        payload["transverse_mm"] = payload["transverse_mm"][:-1]
        payload["calibrated"] = payload["calibrated"][:-1]
        rejected = self.client.put(
            f"{self.api}/transverse", json={"revision": 1, "calibration": payload}
        )
        self.assertEqual(rejected.status_code, 400)


if __name__ == "__main__":
    unittest.main()
