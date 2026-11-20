import io
import json
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

from rollscope.analysis import SECTION_TITLES, AnalysisContext
from rollscope.capture_codec import capture_bytes, capture_hash, read_capture
from rollscope.demo import build_demo
from rollscope.domain import Classification, StationWindow
from rollscope.grid import Support
from rollscope.http import create_app
from rollscope.lineage import station_lineage
from rollscope.records import ReviewWindow
from rollscope.ruler import Ruler, evaluate_ruler
from rollscope.settings import AnalysisSettings
from rollscope.window_parts import MAX_PART_BYTES, build_partition
from rollscope.window_review import WindowCells, review_window, source_rows


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


def full_demo_window():
    capture = build_demo("cold-rolled")
    return ReviewWindow(
        "a" * 16, "Full window", "synthetic", Classification.OPEN, StationWindow.full(capture)
    ), capture


def demo_record(capture, revision=0):
    return SimpleNamespace(
        sha256="b" * 64, id="c" * 16, state=SimpleNamespace(revision=revision), capture=capture
    )


class WindowPartitioning(unittest.TestCase):
    def setUp(self):
        self.window, self.capture = full_demo_window()
        self.context = AnalysisContext(self.capture, AnalysisSettings.defaults())
        self.record = demo_record(self.capture)

    def test_every_part_respects_the_handoff_limit_and_covers_the_window(self):
        _, parts = build_partition(self.record, self.context, self.window)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(part.bytes, MAX_PART_BYTES)
            with ZipFile(io.BytesIO(part.payload)) as archive:
                self.assertEqual(archive.read("manifest.json")[:1], b"{")
                partition = json.loads(archive.read("partition.json"))
                self.assertEqual(partition["part_count"], len(parts))
                self.assertEqual(partition["max_part_bytes"], MAX_PART_BYTES)
                self.assertEqual(partition["window_id"], self.window.id)
        bounds = [(part.row_start, part.row_stop) for part in parts]
        cells = WindowCells.take(self.context, self.window.region)
        self.assertEqual(bounds[0][0], cells.rows.start)
        self.assertEqual(bounds[-1][1], cells.rows.stop)
        for earlier, later in zip(bounds, bounds[1:]):
            self.assertEqual(earlier[1], later[0])

    def test_parts_are_self_contained_and_reassemble_into_the_window(self):
        cells = WindowCells.take(self.context, self.window.region)
        full_lines = {row["source_line"] for row in source_rows(self.context, cells.rows)}
        full_gaps = {
            (gap["first_row"], gap["last_row"])
            for gap in review_window(self.context, self.window)["gaps"]
        }
        _, parts = build_partition(self.record, self.context, self.window)
        collected_lines = set()
        collected_gaps = set()
        grids = []
        for part in parts:
            with ZipFile(io.BytesIO(part.payload)) as archive:
                names = set(archive.namelist())
                self.assertIn("station-cells.npz", names)
                self.assertIn("source-samples.npz", names)
                self.assertIn("source-lines.csv", names)
                self.assertIn("review-sheet.png", names)
                with np.load(io.BytesIO(archive.read("station-cells.npz"))) as data:
                    grids.append(data["reflectance"])
                for row in self._csv_rows(archive, "source-lines.csv"):
                    collected_lines.add(int(row["source_line"]))
                for gap in self._csv_rows(archive, "gaps.csv"):
                    collected_gaps.add((int(gap["first_row"]), int(gap["last_row"])))
        np.testing.assert_array_equal(np.concatenate(grids), cells.values)
        self.assertEqual(collected_lines, full_lines)
        self.assertEqual(collected_gaps, full_gaps)

    def test_spot_footprints_travel_whole_where_they_fit(self):
        _, parts = build_partition(self.record, self.context, self.window)
        self.assertTrue(all(not part.split_spot_ids for part in parts))
        seen = []
        for part in parts:
            with ZipFile(io.BytesIO(part.payload)) as archive:
                seen.extend(
                    int(row["spot_id"]) for row in self._csv_rows(archive, "spots.csv")
                )
        expected = sorted(spot.id for spot in self.context.spots.spots)
        self.assertEqual(sorted(seen), expected)

    def test_a_spot_that_must_be_cut_keeps_one_identity_across_parts(self):
        labels = np.array(self.context.spots.labels, copy=True)
        labels[labels >= 1] += 1
        labels[200:420, 40:120] = 1
        tall = replace(
            self.context.spots.spots[0],
            id=1,
            pixel_count=220 * 80,
            partial=False,
            row_slice=slice(200, 420),
            column_slice=slice(40, 120),
        )
        shifted = tuple(replace(spot, id=spot.id + 1) for spot in self.context.spots.spots)
        self.context.spots = replace(
            self.context.spots, labels=labels, spots=(tall,) + shifted
        )
        from rollscope import window_parts

        original = (window_parts.MAX_PART_BYTES, window_parts.MAX_WINDOW_PARTS)
        window_parts.MAX_PART_BYTES = 400 * 1024
        window_parts.MAX_WINDOW_PARTS = 64
        try:
            _, parts = build_partition(self.record, self.context, self.window)
        finally:
            window_parts.MAX_PART_BYTES, window_parts.MAX_WINDOW_PARTS = original
        carriers = {
            spot_id: part.index
            for part in parts
            for spot_id in part.split_spot_ids
        }
        self.assertIn(1, carriers)
        carrying = [part for part in parts if 1 in part.split_spot_ids]
        self.assertGreaterEqual(len(carrying), 2)
        for part in carrying:
            with ZipFile(io.BytesIO(part.payload)) as archive:
                rows = self._csv_rows(archive, "spots.csv")
                match = next(row for row in rows if int(row["spot_id"]) == 1)
                self.assertEqual(match["spot_split"], "True")
                self.assertEqual(int(match["spot_part_index"]), part.index)
                self.assertEqual(
                    json.loads(match["spot_observation_parts"]),
                    sorted(item.index for item in carrying),
                )

    @staticmethod
    def _csv_rows(archive, name):
        import csv

        return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))


class WindowPartitionRoutes(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.client = create_app(self.temporary.name).test_client()
        ref = self.client.get("/api/captures").json["captures"][0]["id"]
        self.api = f"/api/captures/{ref}"
        overview = self.client.get(self.api).json["overview"]
        created = self.client.post(f"{self.api}/windows", json={
            "revision": 0,
            "label": "Handoff window",
            "note": "route check",
            "status": "open",
            "region": {
                "station_start_mm": overview["station_start_mm"],
                "station_end_mm": overview["station_end_mm"],
                "transverse_start_mm": overview["transverse_start_mm"],
                "transverse_end_mm": overview["transverse_end_mm"],
            },
        })
        self.assertEqual(created.status_code, 201, created.json)
        self.window_id = created.json["window"]["id"]

    def test_part_plan_and_downloads_match_and_full_zip_remains(self):
        plan = self.client.get(f"{self.api}/windows/{self.window_id}/parts?revision=1")
        self.assertEqual(plan.status_code, 200)
        document = plan.json
        self.assertGreater(len(document["parts"]), 1)
        for entry in document["parts"]:
            self.assertLessEqual(entry["bytes"], document["max_part_bytes"])
            response = self.client.get(
                f"{self.api}/windows/{self.window_id}/parts/{entry['part_index']}?revision=1"
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.data), entry["bytes"])
            self.assertIn(
                f"part{entry['part_index']:02d}-of-{len(document['parts']):02d}.zip",
                response.headers["Content-Disposition"],
            )
        full = self.client.get(f"{self.api}/windows/{self.window_id}/export?revision=1")
        self.assertEqual(full.status_code, 200)
        self.assertGreater(len(full.data), max(part["bytes"] for part in document["parts"]))

    def test_unknown_part_and_bad_index_are_rejected(self):
        self.assertEqual(
            self.client.get(f"{self.api}/windows/{self.window_id}/parts/99?revision=1").status_code,
            404,
        )
        for bad in ("0", "abc", "-3"):
            self.assertEqual(
                self.client.get(
                    f"{self.api}/windows/{self.window_id}/parts/{bad}?revision=1"
                ).status_code,
                400,
            )


if __name__ == "__main__":
    unittest.main()
