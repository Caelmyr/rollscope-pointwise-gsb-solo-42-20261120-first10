import io
import json
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from rollscope.analysis import SECTION_TITLES, AnalysisContext
from rollscope.capture_codec import capture_bytes, capture_hash, read_capture
from rollscope.demo import build_demo
from rollscope.domain import Capture, CaptureMetadata, Geometry, UserInputError
from rollscope.grid import Support
from rollscope.http import create_app
from rollscope.lineage import station_lineage
from rollscope.marks import locate_shift
from rollscope.ruler import Ruler, evaluate_ruler
from rollscope.settings import AnalysisSettings, MarkSettings


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


def two_pitch_capture():
    rows, columns = 1600, 200
    increments = np.full(rows, 4, dtype=np.int64)
    increments[0] = 0
    increments[906] = 160  # transport jump leaves stations ~652–680 mm unsampled
    ticks = np.cumsum(increments)
    seconds = np.cumsum(np.full(rows, 0.004))
    stations = ticks * 0.18
    reflectance = np.full((rows, columns), 0.05)
    reflectance[:, 20:180] = 0.6
    band = np.zeros(columns)
    band[125:175] = 1.0
    positions = [40 + 60 * k for k in range(11)]  # 60 mm pitch up to 640 mm
    positions += [740 + 90 * k for k in range(5)]  # 90 mm pitch after the changeover
    positions.remove(920)  # one genuinely missing mark in the second segment
    for center in positions:
        profile = np.exp(-0.5 * ((stations - center) / 1.3) ** 2)
        reflectance -= 0.35 * profile[:, None] * band[None, :]
    raw = np.rint(800 + 40000 * reflectance).clip(0, 65535).astype(np.uint16)
    dark = np.full((3, columns), 800, dtype=np.uint16)
    white = np.full((3, columns), 40800, dtype=np.uint16)
    geometry = Geometry(0.18, 0.28, 0.0, 0.0)
    metadata = CaptureMetadata(
        "Two-pitch marks", "Test strip", "Synthetic two-pitch mark train.", True, geometry
    )
    return Capture.build(raw, ticks, seconds, dark, white, metadata)


class PitchShiftSearch(unittest.TestCase):
    def test_unique_transition_either_spacing_order(self):
        stations = np.array([0, 60, 120, 180, 330, 420, 510], dtype=float)
        assignment, plausible, misfits = locate_shift(stations, (60, 90), 100)
        self.assertEqual(assignment, (60, 90))
        self.assertEqual(plausible, [3])
        self.assertEqual(misfits, 0)
        reversed_assignment, reversed_plausible, _ = locate_shift(stations, (90, 60), 100)
        self.assertEqual(reversed_assignment, (60, 90))
        self.assertEqual(reversed_plausible, [3])

    def test_tied_splits_widen_the_range(self):
        stations = np.array([0, 60, 120, 180, 270, 360], dtype=float)
        assignment, plausible, misfits = locate_shift(stations, (60, 90), 90)
        self.assertEqual(assignment, (60, 90))
        self.assertEqual(plausible, [2, 3])
        self.assertEqual(misfits, 0)

    def test_missing_mark_inside_segment_does_not_hide_it(self):
        stations = np.array([0, 60, 180, 240, 330, 420, 510], dtype=float)
        assignment, plausible, misfits = locate_shift(stations, (60, 90), 100)
        self.assertEqual(assignment, (60, 90))
        self.assertEqual(plausible, [2, 3])
        self.assertEqual(misfits, 0)

    def test_minimum_segment_can_make_shift_unplaceable(self):
        stations = np.array([0, 60, 120, 210, 300], dtype=float)
        self.assertIsNone(locate_shift(stations, (60, 90), 400))
        self.assertIsNone(locate_shift(stations[:1], (60, 90), 0))


class MarkSettingsValidation(unittest.TestCase):
    def test_defaults_keep_single_pitch(self):
        settings = MarkSettings.parse({})
        self.assertIsNone(settings.second_spacing_mm)
        self.assertEqual(settings.minimum_segment_mm, 240)

    def test_two_segment_validation(self):
        settings = MarkSettings.parse({"second_spacing_mm": 90, "minimum_segment_mm": 300})
        self.assertEqual(settings.second_spacing_mm, 90)
        self.assertEqual(settings.minimum_segment_mm, 300)
        with self.assertRaises(UserInputError):
            MarkSettings.parse({"second_spacing_mm": 60})
        with self.assertRaises(UserInputError):
            MarkSettings.parse({"second_spacing_mm": 90, "minimum_segment_mm": 50})

    def test_blank_second_spacing_keeps_single_mode(self):
        settings = MarkSettings.parse({"second_spacing_mm": None, "minimum_segment_mm": None})
        self.assertIsNone(settings.second_spacing_mm)
        self.assertEqual(settings.minimum_segment_mm, 240)


class TwoPitchMarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = two_pitch_capture()

    def marks_context(self, **marks_values):
        settings = AnalysisSettings.defaults().with_section("marks", marks_values)
        return AnalysisContext(self.capture, settings)

    def test_two_segment_locates_shift_range_over_gap(self):
        report = self.marks_context(
            expected_spacing_mm=60, second_spacing_mm=90, minimum_segment_mm=240
        ).marks
        self.assertEqual(report.mode, "two_segment")
        shift = report.shift
        self.assertIsNotNone(shift)
        self.assertEqual(shift.first_spacing_mm, 60)
        self.assertEqual(shift.second_spacing_mm, 90)
        # The true changeover at 650 mm and the unsampled span both lie inside the range.
        self.assertLess(shift.from_mm, 650)
        self.assertGreater(shift.to_mm, 650)
        self.assertLessEqual(shift.from_mm, 651.6)
        self.assertGreaterEqual(shift.to_mm, 680.4)

        summary = report.summary()
        self.assertEqual(summary["mode"], "two_segment")
        self.assertEqual(summary["shift_status"], "located")
        # Only the genuinely missing 90 mm mark is counted.
        self.assertEqual(summary["missed_candidates"], 1)
        multiples = [i for i in report.intervals if i.classification == "multiple_spacing"]
        self.assertEqual(len(multiples), 1)
        self.assertEqual(multiples[0].expected_mm, 90)
        self.assertEqual(multiples[0].inferred_missing_marks, 1)
        # The changeover intervals stay unjudged instead of becoming missed prints.
        transitions = [i for i in report.intervals if i.classification == "shift_transition"]
        self.assertEqual(len(transitions), shift.transition_intervals)
        self.assertTrue(all(i.inferred_missing_marks == 0 for i in transitions))
        self.assertTrue(all(i.expected_mm is None for i in transitions))
        self.assertFalse(any(i.classification == "irregular" for i in report.intervals))
        first = summary["first_segment_spacing_mm"]
        second = summary["second_segment_spacing_mm"]
        self.assertAlmostEqual(first["median"], 60, delta=1)
        self.assertAlmostEqual(second["median"], 90, delta=1)

        groups = [mark.pitch_group for mark in report.marks]
        self.assertEqual(groups[0], "first")
        self.assertEqual(groups[-1], "second")
        self.assertIn("shift", groups)

    def test_single_pitch_still_checks_the_same_capture(self):
        report = self.marks_context(expected_spacing_mm=60).marks
        self.assertEqual(report.mode, "single_pitch")
        self.assertIsNone(report.shift)
        summary = report.summary()
        self.assertEqual(summary["mode"], "single_pitch")
        # Fixed 60 mm checking misreads the changeover region as missed marks.
        self.assertEqual(summary["missed_candidates"], 2)
        self.assertTrue(any(i.classification == "irregular" for i in report.intervals))


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

    def test_marks_two_segment_mode_and_back(self):
        switched = self.client.post(f"{self.api}/settings", json={
            "revision": 0,
            "section": "marks",
            "values": {"second_spacing_mm": 90, "minimum_segment_mm": 240},
        })
        self.assertEqual(switched.status_code, 200, switched.json)
        analysis = self.client.get(f"{self.api}/analysis/marks?revision=1")
        self.assertEqual(analysis.status_code, 200, analysis.json)
        self.assertEqual(analysis.json["summary"]["mode"], "two_segment")
        keys = [table["key"] for table in analysis.json["tables"]]
        self.assertIn("shift", keys)
        restored = self.client.post(f"{self.api}/settings", json={
            "revision": 1,
            "section": "marks",
            "values": {"second_spacing_mm": None},
        })
        self.assertEqual(restored.status_code, 200, restored.json)
        again = self.client.get(f"{self.api}/analysis/marks?revision=2")
        self.assertEqual(again.status_code, 200, again.json)
        self.assertEqual(again.json["summary"]["mode"], "single_pitch")

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
