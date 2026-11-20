import io
import tempfile
import unittest
from zipfile import ZipFile

from rollscope.analysis import AnalysisContext
from rollscope.demo import build_demo
from rollscope.http import create_app
from rollscope.settings import AnalysisSettings
from rollscope.trim import (
    CUT_LIMIT,
    MAX_REMOVED_SPOTS,
    TrimObjective,
    TrimPlanner,
    TrimRequest,
    evaluate_trim,
)


class TrimPlanning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = build_demo("cold-rolled")
        cls.context = AnalysisContext(cls.capture, AnalysisSettings.defaults())
        cls.planner = TrimPlanner(cls.context)

    def request(self, **overrides):
        values = {
            "margin_mm": 3.0,
            "minimum_length_mm": 50.0,
            "retain_start_mm": None,
            "retain_end_mm": None,
            "spot_ids": [],
        }
        values.update(overrides)
        return TrimRequest.parse(values)

    def test_cuts_land_on_sampled_stations_and_enclose_footprints(self):
        request = self.request(spot_ids=[32])
        plan = self.planner.plan(TrimObjective.LONGEST, request)
        self.assertTrue(plan.feasible, plan.conflicts)
        region, = plan.cut_regions
        self.assertEqual(plan.cut_count, CUT_LIMIT)
        spot = next(item for item in plan.selected_spots if item.spot_id == 32)
        # Cuts snap outward onto sampled rows, so they bracket the isolation interval.
        self.assertLessEqual(region.start_row, self.planner.snap_outward(spot.isolation_start_mm, "left"))
        self.assertGreaterEqual(region.end_row, self.planner.snap_outward(spot.isolation_end_mm, "right"))
        self.assertLessEqual(region.start_mm, spot.isolation_start_mm + self.planner.pitch + 1e-9)
        self.assertGreaterEqual(region.end_mm, spot.isolation_end_mm - self.planner.pitch - 1e-9)
        for cut in (region.start_cut, region.end_cut):
            self.assertGreater(cut.valid_cells, 0)
            self.assertNotEqual(cut.support, "gap")

    def test_margin_expands_the_isolation_interval(self):
        narrow = self.planner.plan(TrimObjective.LONGEST, self.request(margin_mm=1.0, spot_ids=[32]))
        wide = self.planner.plan(TrimObjective.LONGEST, self.request(margin_mm=8.0, spot_ids=[32]))
        self.assertGreater(
            wide.cut_regions[0].length_mm, narrow.cut_regions[0].length_mm
        )

    def test_merged_spots_share_one_cut_region(self):
        plan = self.planner.plan(TrimObjective.LONGEST, self.request(spot_ids=[1, 2]))
        self.assertTrue(plan.feasible)
        self.assertEqual(len(plan.cut_regions), 1)
        self.assertEqual(plan.cut_regions[0].spot_ids, (1, 2))

    def test_separate_intervals_exceed_two_cut_window(self):
        plan = self.planner.plan(
            TrimObjective.LONGEST, self.request(margin_mm=1.0, spot_ids=[1, 32])
        )
        self.assertFalse(plan.feasible)
        self.assertTrue(any(item["code"] == "cut_window_exceeded" for item in plan.conflicts))

    def test_more_than_three_spots_rejected(self):
        with self.assertRaisesRegex(Exception, "At most 3"):
            self.request(spot_ids=[1, 2, 3, 4])

    def test_unknown_spot_is_a_conflict_not_a_crash(self):
        plan = self.planner.plan(
            TrimObjective.LONGEST, self.request(spot_ids=[999])
        )
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.conflicts[0]["code"], "unknown_spot")

    def test_minimum_length_marks_short_remnants_not_shippable(self):
        plan = self.planner.plan(
            TrimObjective.LONGEST, self.request(minimum_length_mm=400.0, spot_ids=[32])
        )
        self.assertTrue(any(not segment.shippable for segment in plan.segments))
        self.assertTrue(all(segment.length_mm >= 0 for segment in plan.segments))

    def test_retained_sample_must_stay_inside_one_remnant(self):
        ok = self.planner.plan(
            TrimObjective.LONGEST,
            self.request(spot_ids=[32], retain_start_mm=1800.0, retain_end_mm=1850.0),
        )
        self.assertTrue(ok.feasible)
        self.assertTrue(any(segment.contains_retain for segment in ok.segments))

        overlapping = self.planner.plan(
            TrimObjective.LONGEST,
            self.request(spot_ids=[32], retain_start_mm=1715.0, retain_end_mm=1717.0),
        )
        self.assertFalse(overlapping.feasible)
        self.assertTrue(
            any(item["code"] == "retain_inside_cutout" for item in overlapping.conflicts)
        )

    def test_two_objectives_can_disagree(self):
        result = evaluate_trim(self.context, {
            "margin_mm": 3.0,
            "minimum_length_mm": 50.0,
            "retain_start_mm": None,
            "retain_end_mm": None,
            "spot_ids": [],
        })
        self.assertTrue(result["objectives_differ"])
        longest = result["longest_segment"]
        retained = result["max_retained"]
        self.assertGreaterEqual(
            longest["longest_segment_mm"], retained["longest_segment_mm"]
        )
        self.assertGreaterEqual(
            retained["retained_length_mm"], longest["retained_length_mm"]
        )
        self.assertTrue(longest["selected_spots"])
        self.assertLessEqual(len(longest["selected_spots"]), MAX_REMOVED_SPOTS)

    def test_length_conservation(self):
        plan = self.planner.plan(TrimObjective.LONGEST, self.request(spot_ids=[32]))
        accounted = plan.removed_length_mm + plan.retained_length_mm
        self.assertAlmostEqual(accounted, plan.total_length_mm, places=6)
        self.assertAlmostEqual(
            sum(segment.length_mm for segment in plan.segments),
            plan.retained_length_mm,
            places=6,
        )

    def test_partial_spots_produce_warnings(self):
        plan = self.planner.plan(TrimObjective.LONGEST, self.request(spot_ids=[1]))
        self.assertTrue(plan.warnings)

    def test_cluster_enumeration_matches_brute_force(self):
        from itertools import combinations

        request = self.request()
        ids = sorted(self.planner.spots)
        brute_force = set()
        for size in (1, 2, 3):
            for combo in combinations(ids, size):
                plan = self.planner.plan(
                    TrimObjective.LONGEST,
                    TrimRequest(
                        request.margin_mm,
                        request.minimum_length_mm,
                        request.retain_start_mm,
                        request.retain_end_mm,
                        combo,
                    ),
                )
                if plan.feasible:
                    brute_force.add(
                        tuple((region.start_row, region.end_row) for region in plan.cut_regions)
                    )
        clustered = set()
        for combo in self.planner.feasible_subsets(request):
            if not combo:
                continue
            plan = self.planner.plan(
                TrimObjective.LONGEST,
                TrimRequest(
                    request.margin_mm,
                    request.minimum_length_mm,
                    request.retain_start_mm,
                    request.retain_end_mm,
                    combo,
                ),
            )
            if plan.feasible:
                clustered.add(
                    tuple((region.start_row, region.end_row) for region in plan.cut_regions)
                )
        self.assertEqual(brute_force, clustered)


class TrimWorkspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.app = create_app(self.temporary.name)
        self.client = self.app.test_client()
        self.ref = self.client.get("/api/captures").json["captures"][0]["id"]
        self.api = f"/api/captures/{self.ref}"

    def options(self, revision=0, **overrides):
        payload = {
            "revision": revision,
            "margin_mm": 3.0,
            "minimum_length_mm": 50.0,
            "retain_start_mm": None,
            "retain_end_mm": None,
            "spot_ids": [],
        }
        payload.update(overrides)
        return self.client.post(self.api + "/trim/options", json=payload)

    def test_options_preview_and_confirmation(self):
        result = self.options().json
        self.assertTrue(result["objectives_differ"])
        chosen = result["longest_segment"]
        preview = self.client.post(self.api + "/trim/preview.png", json={
            "revision": 0,
            "objective": "longest_segment",
            "margin_mm": 3.0,
            "minimum_length_mm": 50.0,
            "retain_start_mm": None,
            "retain_end_mm": None,
            "spot_ids": [item["spot_id"] for item in chosen["selected_spots"]],
        })
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.data[:8], b"\x89PNG\r\n\x1a\n")
        confirmed = self.client.post(self.api + "/trim/plans", json={
            "revision": 0,
            "label": "Downtime cut set",
            "objective": "longest_segment",
            "margin_mm": 3.0,
            "minimum_length_mm": 50.0,
            "retain_start_mm": None,
            "retain_end_mm": None,
            "spot_ids": [item["spot_id"] for item in chosen["selected_spots"]],
        })
        self.assertEqual(confirmed.status_code, 201, confirmed.json)
        self.assertEqual(len(confirmed.json["state"]["trim_plans"]), 1)

    def test_conflicting_plan_cannot_be_confirmed(self):
        rejected = self.client.post(self.api + "/trim/plans", json={
            "revision": 0,
            "label": "Impossible",
            "objective": "longest_segment",
            "margin_mm": 1.0,
            "minimum_length_mm": 50.0,
            "retain_start_mm": None,
            "retain_end_mm": None,
            "spot_ids": [1, 32],
        })
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("cannot be confirmed", rejected.json["error"])

    def test_saved_plan_reopens_and_tracks_settings_drift(self):
        chosen = self.options().json["max_retained"]
        saved = self.client.post(self.api + "/trim/plans", json={
            "revision": 0,
            "label": "Keep most material",
            "objective": "max_retained",
            "margin_mm": 3.0,
            "minimum_length_mm": 50.0,
            "retain_start_mm": None,
            "retain_end_mm": None,
            "spot_ids": [item["spot_id"] for item in chosen["selected_spots"]],
        }).json["trim_plan"]
        opened = self.client.get(self.api + "/trim/plans/" + saved["id"])
        self.assertEqual(opened.status_code, 200)
        self.assertTrue(opened.json["settings_match"])

        self.client.post(self.api + "/settings", json={
            "revision": 1, "section": "spots", "values": {"contrast": 0.28},
        })
        drifted = self.client.get(self.api + "/trim/plans/" + saved["id"]).json
        self.assertFalse(drifted["settings_match"])
        self.assertEqual(drifted["saved"]["id"], saved["id"])

    def test_export_and_project_restore_roundtrip(self):
        chosen = self.options().json["longest_segment"]
        saved = self.client.post(self.api + "/trim/plans", json={
            "revision": 0,
            "label": "Exported set",
            "objective": "longest_segment",
            "margin_mm": 3.0,
            "minimum_length_mm": 50.0,
            "retain_start_mm": None,
            "retain_end_mm": None,
            "spot_ids": [item["spot_id"] for item in chosen["selected_spots"]],
        }).json["trim_plan"]
        package = self.client.get(self.api + "/trim/plans/" + saved["id"] + "/export")
        self.assertEqual(package.status_code, 200)
        with ZipFile(io.BytesIO(package.data)) as archive:
            names = archive.namelist()
            self.assertIn("trim-overview.png", names)
            self.assertIn("cut-positions.csv", names)
            self.assertIn("remnant-segments.csv", names)
            self.assertIn("trim-plan.json", names)

        backup = self.client.get(self.api + "/project").data
        restored = self.client.post(
            "/api/captures/restore",
            data={"project": (io.BytesIO(backup), "roundtrip.zip")},
        )
        self.assertEqual(restored.status_code, 201)
        new_id = restored.json["id"]
        state = self.client.get(f"/api/captures/{new_id}").json["state"]
        self.assertEqual(len(state["trim_plans"]), 1)
        plan = state["trim_plans"][0]
        self.assertEqual(plan["spot_ids"], saved["spot_ids"])
        reopened = self.client.get(f"/api/captures/{new_id}/trim/plans/{plan['id']}")
        self.assertEqual(reopened.status_code, 200)


if __name__ == "__main__":
    unittest.main()
