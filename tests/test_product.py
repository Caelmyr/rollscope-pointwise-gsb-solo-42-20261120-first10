import io
import json
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from rollscope.analysis import SECTION_TITLES, AnalysisContext, AnalysisPresenter
from rollscope.capture_codec import capture_bytes, capture_hash, read_capture
from rollscope.demo import build_demo
from rollscope.grid import Support
from rollscope.http import create_app
from rollscope.lineage import station_lineage
from rollscope.ruler import Ruler, evaluate_ruler
from rollscope.settings import AnalysisSettings
from rollscope.streaks import classify_alignment


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


class StreakTrajectories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = AnalysisContext(build_demo("cold-rolled"), AnalysisSettings.defaults())

    def test_alignment_rules(self):
        self.assertEqual(classify_alignment(8.0, 0.4, 10.0, 0.9), "material")
        self.assertEqual(classify_alignment(0.4, 8.0, 10.0, 0.9), "fixed")
        self.assertEqual(classify_alignment(0.4, 0.4, 0.3, 0.9), "unclear")
        self.assertEqual(classify_alignment(8.0, 0.4, 10.0, 0.3), "unclear")
        self.assertEqual(classify_alignment(4.0, 4.0, 8.0, 0.9), "unclear")

    def test_material_streak_follows_strip_weave(self):
        paths = self.context.streaks.path_table()
        material = [path for path in paths if path["alignment"] == "material"]
        self.assertTrue(material)
        for path in material:
            self.assertGreater(path["strip_motion_mm"], 1.0)
            self.assertGreater(path["drift_mm"], 0.5)

    def test_camera_locked_streak_stays_on_column(self):
        paths = self.context.streaks.path_table()
        fixed = [path for path in paths if path["alignment"] == "fixed"]
        self.assertTrue(fixed)
        for path in fixed:
            self.assertLess(path["drift_mm"], 0.5)
            self.assertGreater(path["strip_motion_mm"], 1.0)

    def test_paths_break_at_acquisition_gaps(self):
        gaps = self.context.coverage.gaps
        self.assertTrue(gaps)
        for path in self.context.streaks.paths:
            for gap in gaps:
                self.assertFalse(
                    path.from_mm < gap.from_mm and path.to_mm > gap.to_mm,
                    f"path {path.id} spans acquisition gap {gap.id}",
                )

    def test_paths_reference_fixed_column_streaks(self):
        report = self.context.streaks
        streak_ids = {streak.id for streak in report.streaks}
        members = set()
        for path in report.paths:
            self.assertTrue(path.members)
            members.update(path.members)
        self.assertTrue(members)
        self.assertLessEqual(members, streak_ids)

    def test_streak_section_reports_paths_and_column_tracks(self):
        output = AnalysisPresenter(self.context).output("streaks")
        self.assertEqual([table["key"] for table in output["tables"]], ["paths", "streaks"])
        counts = output["summary"]["paths"]
        self.assertGreaterEqual(counts["material"], 1)
        self.assertGreaterEqual(counts["fixed"], 1)
        self.assertEqual(
            sum(counts.values()), len(self.context.streaks.paths)
        )


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
