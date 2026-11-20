import io
import json
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from rollscope.alignment import build_report, leg_assignment, parse_log_text
from rollscope.analysis import SECTION_TITLES, AnalysisContext
from rollscope.capture_codec import capture_bytes, capture_hash, read_capture
from rollscope.demo import build_demo
from rollscope.domain import UserInputError
from rollscope.grid import Support
from rollscope.http import create_app
from rollscope.lineage import station_lineage
from rollscope.records import LogAlignment
from rollscope.ruler import Ruler, evaluate_ruler
from rollscope.settings import AnalysisSettings


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


LOG_TEXT = """# kind, start, end, note
试压, 0, 0, known test A
停机, 0.5, 2.5, long stop
点检, 1.5, 1.5, mid check
试压, 3, 3, known test B
换辊, 1, 1, roll change after rollback
试压, 2, 2, known test C
"""

CONSISTENT_ANCHORS = [
    {"event_id": "e1", "station_start_mm": 1246, "station_end_mm": 1254},
    {"event_id": "e4", "station_start_mm": 1546, "station_end_mm": 1554},
    {"event_id": "e5", "station_start_mm": 1716, "station_end_mm": 1724},
    {"event_id": "e6", "station_start_mm": 1816, "station_end_mm": 1824},
]


class Alignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = build_demo("cold-rolled")
        cls.context = AnalysisContext(cls.capture, AnalysisSettings.defaults())
        cls.events, _ = parse_log_text(LOG_TEXT)

    def report(self, anchors, certainty=25):
        alignment = LogAlignment.build(
            {"log_text": LOG_TEXT, "anchors": anchors, "certainty_mm": certainty},
            self.events,
            self.capture,
        )
        return build_report(self.capture, self.context.grid, self.context.coverage.gaps, alignment)

    def test_log_parsing_splits_legs_at_rollback(self):
        events, warnings = parse_log_text(LOG_TEXT)
        self.assertEqual(warnings, [])
        self.assertEqual(len(events), 6)
        self.assertEqual(leg_assignment(events), [0, 0, 0, 0, 1, 1])
        self.assertEqual(events[0].kind, "试压")
        self.assertEqual(events[4].start_s, 1.0)

    def test_log_parsing_accepts_datetime_and_clock_forms(self):
        events, _ = parse_log_text(
            "停机, 2024-03-15 08:30:00, 2024-03-15 08:31:00\n试压, 08:32, 08:33:30.5"
        )
        self.assertEqual(events[0].end_s - events[0].start_s, 60.0)
        self.assertEqual(events[1].start_s, 8 * 3600 + 32 * 60)
        self.assertAlmostEqual(events[1].end_s, 8 * 3600 + 33 * 60 + 30.5)

    def test_log_parsing_rejects_reversed_and_unreadable_records(self):
        with self.assertRaises(UserInputError):
            parse_log_text("试压, 10, 9")
        with self.assertRaises(UserInputError):
            parse_log_text("试压, not-a-time, 10")
        with self.assertRaises(UserInputError):
            parse_log_text("试压, 1")
        with self.assertRaises(UserInputError):
            parse_log_text("# only a comment")

    def test_envelopes_distinguish_confident_and_ambiguous(self):
        report = self.report(CONSISTENT_ANCHORS)
        self.assertEqual(report["rollback_count"], 1)
        self.assertEqual([leg["status"] for leg in report["legs"]], ["aligned", "aligned"])
        placements = {item["event_id"]: item for item in report["placements"]}
        point = placements["e3"]
        self.assertEqual(point["status"], "confident")
        self.assertAlmostEqual(point["station_mm"], 1400.0, delta=4.0)
        self.assertLessEqual(point["width_mm"], 25)
        stop = placements["e2"]
        self.assertEqual(stop["status"], "ambiguous")
        self.assertLessEqual(stop["station_start_mm"], 1300.0)
        self.assertGreaterEqual(stop["station_end_mm"], 1500.0)
        self.assertTrue(stop["crosses_gap"])
        self.assertEqual(placements["e1"]["status"], "confident")
        self.assertEqual(placements["e6"]["status"], "confident")
        boundary = report["boundaries"][0]
        self.assertEqual(boundary["status"], "consistent")
        self.assertIsNone(boundary["suspect_leg"])
        self.assertEqual(len(report["gaps"]), 2)

    def test_boundary_mismatch_names_suspect_side(self):
        anchors = [
            {"event_id": "e1", "station_start_mm": 1246, "station_end_mm": 1254},
            {"event_id": "e4", "station_start_mm": 1546, "station_end_mm": 1554},
            {"event_id": "e5", "station_start_mm": 1500, "station_end_mm": 1508},
            {"event_id": "e6", "station_start_mm": 1800, "station_end_mm": 1808},
        ]
        boundary = self.report(anchors)["boundaries"][0]
        self.assertEqual(boundary["status"], "inconsistent")
        self.assertGreater(boundary["station_drop_mm"], 30)
        self.assertEqual(boundary["suspect_leg"], 1)
        self.assertEqual(boundary["suspect_title"], "After rollback")

    def test_conflicting_and_underdetermined_legs(self):
        conflicting = self.report([
            {"event_id": "e1", "station_start_mm": 1246, "station_end_mm": 1254},
            {"event_id": "e4", "station_start_mm": 1210, "station_end_mm": 1218},
            {"event_id": "e5", "station_start_mm": 1716, "station_end_mm": 1724},
            {"event_id": "e6", "station_start_mm": 1816, "station_end_mm": 1824},
        ])
        self.assertEqual(conflicting["legs"][0]["status"], "conflicting")
        self.assertEqual(conflicting["legs"][0]["conflict"]["events"], ["e1", "e4"])
        self.assertEqual(conflicting["placements"][0]["status"], "unplaced")
        self.assertEqual(conflicting["boundaries"][0]["status"], "unverifiable")
        sparse = self.report(CONSISTENT_ANCHORS[:3])
        self.assertEqual(sparse["legs"][1]["status"], "underdetermined")
        self.assertEqual(sparse["placements"][4]["status"], "unplaced")

    def test_anchor_validation_uses_capture_range(self):
        with self.assertRaises(UserInputError):
            self.report([
                {"event_id": "e1", "station_start_mm": 100, "station_end_mm": 200},
            ])
        with self.assertRaises(UserInputError):
            self.report([
                {"event_id": "e9", "station_start_mm": 1246, "station_end_mm": 1254},
            ])


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

    def test_alignment_lifecycle_survives_project_restore(self):
        parsed = self.client.post(f"{self.api}/alignment/parse", json={"log_text": LOG_TEXT})
        self.assertEqual(parsed.status_code, 200, parsed.json)
        self.assertEqual(parsed.json["rollback_count"], 1)
        self.assertEqual([event["leg"] for event in parsed.json["events"]], [0, 0, 0, 0, 1, 1])

        empty = self.client.get(f"{self.api}/alignment")
        self.assertEqual(empty.status_code, 200)
        self.assertIsNone(empty.json["alignment"])

        saved = self.client.put(f"{self.api}/alignment", json={
            "revision": 0,
            "log_text": LOG_TEXT,
            "anchors": CONSISTENT_ANCHORS,
            "certainty_mm": 25,
        })
        self.assertEqual(saved.status_code, 200, saved.json)
        report = saved.json["report"]
        self.assertEqual(report["boundaries"][0]["status"], "consistent")
        placements = {item["event_id"]: item for item in report["placements"]}
        self.assertEqual(placements["e3"]["status"], "confident")
        self.assertEqual(placements["e2"]["status"], "ambiguous")
        self.assertEqual(saved.json["state"]["revision"], 1)

        conflict = self.client.put(f"{self.api}/alignment", json={
            "revision": 0,
            "log_text": LOG_TEXT,
            "anchors": CONSISTENT_ANCHORS,
            "certainty_mm": 25,
        })
        self.assertEqual(conflict.status_code, 409)

        fetched = self.client.get(f"{self.api}/alignment")
        self.assertEqual(fetched.json["alignment"]["anchors"][0]["event_id"], "e1")
        self.assertEqual(fetched.json["alignment"]["log_text"], LOG_TEXT.strip())

        archive = self.client.get(f"{self.api}/project").data
        restored = self.client.post("/api/captures/restore", data={
            "project": (io.BytesIO(archive), "aligned.zip"),
        })
        self.assertEqual(restored.status_code, 201, restored.json)
        replica = self.client.get(f"/api/captures/{restored.json['id']}/alignment")
        self.assertEqual(replica.json["alignment"]["log_text"], LOG_TEXT.strip())
        self.assertEqual(replica.json["alignment"]["anchors"], CONSISTENT_ANCHORS)
        self.assertEqual(replica.json["report"]["placements"], report["placements"])

        removed = self.client.delete(f"{self.api}/alignment", json={"revision": 1})
        self.assertEqual(removed.status_code, 200, removed.json)
        self.assertIsNone(self.client.get(f"{self.api}/alignment").json["alignment"])

    def test_alignment_rejects_bad_anchors_and_logs(self):
        reversed_log = self.client.put(f"{self.api}/alignment", json={
            "revision": 0, "log_text": "试压, 10, 9", "anchors": [], "certainty_mm": 25,
        })
        self.assertEqual(reversed_log.status_code, 400)
        outside = self.client.put(f"{self.api}/alignment", json={
            "revision": 0,
            "log_text": LOG_TEXT,
            "anchors": [{"event_id": "e1", "station_start_mm": 10, "station_end_mm": 20}],
            "certainty_mm": 25,
        })
        self.assertEqual(outside.status_code, 400)
        unknown = self.client.put(f"{self.api}/alignment", json={
            "revision": 0,
            "log_text": LOG_TEXT,
            "anchors": [{"event_id": "e9", "station_start_mm": 1246, "station_end_mm": 1254}],
            "certainty_mm": 25,
        })
        self.assertEqual(unknown.status_code, 400)
        self.assertIsNone(self.client.get(f"{self.api}/alignment").json["alignment"])


if __name__ == "__main__":
    unittest.main()
