import io
import json
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from rollscope.analysis import SECTION_TITLES, AnalysisContext
from rollscope.blocks import BlockGrid, _split_edges, reduce_blocks
from rollscope.capture_codec import capture_bytes, capture_hash, read_capture
from rollscope.demo import build_demo
from rollscope.grid import Support
from rollscope.http import create_app
from rollscope.lineage import station_lineage
from rollscope.raster import Layer, RasterSpec, position_mapping, preview_image
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


class BlockReduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = build_demo("cold-rolled")
        cls.context = AnalysisContext(cls.capture, AnalysisSettings.defaults())
        cls.grid = cls.context.grid
        cls.blocks = BlockGrid.make(
            slice(0, cls.grid.shape[0]), slice(0, cls.grid.shape[1]), 300, 120
        )

    def test_split_edges_are_complete_and_near_uniform(self):
        for length, count in ((1156, 300), (320, 120), (10, 3), (7, 7), (100, 1)):
            edges = _split_edges(0, length, count)
            sizes = np.diff(edges)
            self.assertEqual(len(edges), count + 1)
            self.assertEqual(int(edges[0]), 0)
            self.assertEqual(int(edges[-1]), length)
            self.assertTrue((sizes >= 1).all())
            self.assertLessEqual(int(sizes.max() - sizes.min()), 1)

    def test_block_statistics_match_their_source_cells(self):
        facts = reduce_blocks(self.context, Layer.STATION, self.blocks)
        rng = np.random.default_rng(0)
        for _ in range(200):
            br = int(rng.integers(0, self.blocks.block_rows))
            bc = int(rng.integers(0, self.blocks.block_columns))
            r0, r1 = self.blocks.row_bounds(br)
            c0, c1 = self.blocks.column_bounds(bc)
            patch = self.grid.values[r0:r1, c0:c1]
            finite = patch[np.isfinite(patch)]
            if finite.size:
                self.assertAlmostEqual(
                    np.median(finite.astype(np.float64)), facts.median[br, bc], places=5
                )
                self.assertAlmostEqual(finite.min(), facts.minimum[br, bc], places=6)
                self.assertAlmostEqual(finite.max(), facts.maximum[br, bc], places=6)
            support_total = (
                facts.native_cells[br, bc]
                + facts.blend_cells[br, bc]
                + facts.gap_cells[br, bc]
            )
            self.assertEqual(int(support_total), int(facts.total_cells[br, bc]))

    def test_narrow_gaps_survive_as_minority_blocks(self):
        facts = reduce_blocks(self.context, Layer.STATION, self.blocks)
        minority_gap = (facts.gap_fraction() > 0) & (facts.gap_fraction() < 0.5)
        self.assertGreater(int(minority_gap.sum()), 0)
        # The old centre-sample approach would miss every one of these blocks
        # because their own centre source row is not an encoder-gap row.
        missed_by_centre = 0
        for br, bc in np.argwhere(minority_gap):
            r0, r1 = self.blocks.row_bounds(int(br))
            centre_row = (r0 + r1) // 2
            if self.grid.plan.support[centre_row] != Support.GAP:
                missed_by_centre += 1
        self.assertGreater(missed_by_centre, 0)

    def test_contrast_extremes_keep_narrow_spot_evidence(self):
        facts = reduce_blocks(self.context, Layer.SPOTS, self.blocks)
        bright_blocks = np.isfinite(facts.contrast_max) & (facts.contrast_max > 0)
        dark_blocks = np.isfinite(facts.contrast_min) & (facts.contrast_min < 0)
        self.assertGreater(int(bright_blocks.sum()), 50)
        self.assertGreater(int(dark_blocks.sum()), 50)
        self.assertIn(1, facts.spot_ids)

    def test_downsampled_preview_has_one_pixel_per_block(self):
        spec = RasterSpec(Layer.COVERAGE, max_width=120, max_height=300)
        mapping = position_mapping(self.context, spec)
        self.assertTrue(mapping.block_mode)
        image, returned = preview_image(self.context, spec)
        self.assertEqual(image.size, (mapping.width, mapping.height))
        self.assertTrue(returned.block_mode)
        self.assertEqual(returned.as_dict()["block_rows"], mapping.block_grid.block_rows)
        self.assertEqual(returned.as_dict()["block_columns"], mapping.block_grid.block_columns)

    def test_full_resolution_view_keeps_one_to_one_cells(self):
        spec = RasterSpec(Layer.COVERAGE)
        mapping = position_mapping(self.context, spec)
        self.assertFalse(mapping.block_mode)
        image, _ = preview_image(self.context, spec)
        self.assertEqual(image.size, (mapping.width, mapping.height))

    def test_context_caches_block_facts_for_repeated_probes(self):
        first = self.context.block_facts(Layer.STATION, self.blocks)
        second = self.context.block_facts(Layer.STATION, self.blocks)
        self.assertIs(first, second)
        other_layer = self.context.block_facts(Layer.SPOTS, self.blocks)
        self.assertIsNot(first, other_layer)


class BlockProbeAndRegion(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.app = create_app(self.temporary.name)
        self.client = self.app.test_client()
        self.ref = self.client.get("/api/captures").json["captures"][0]["id"]
        self.api = f"/api/captures/{self.ref}"

    def test_region_view_and_probe_share_one_physical_frame(self):
        bounds = (
            "station_start_mm=1450&station_end_mm=1540"
            "&transverse_start_mm=8&transverse_end_mm=80"
        )
        view = self.client.get(f"{self.api}/view?layer=spots&revision=0&{bounds}")
        self.assertEqual(view.status_code, 200)
        self.assertFalse(view.json["block_mode"])
        probe = self.client.get(f"{self.api}/probe?layer=spots&revision=0&x=0.5&y=0.5&{bounds}")
        self.assertEqual(probe.status_code, 200)
        # The probe stays inside the requested frame on every station view.
        for section_layer in ("station", "coverage", "spots"):
            response = self.client.get(
                f"{self.api}/view?layer={section_layer}&revision=0&{bounds}"
            )
            self.assertEqual(response.json["x_from_mm"], view.json["x_from_mm"])
            self.assertEqual(response.json["y_from"], view.json["y_from"])
        self.assertIn("station_mm", probe.json)

    def test_region_requires_complete_bounds_and_valid_frame(self):
        partial = self.client.get(
            f"{self.api}/view?layer=station&revision=0&station_start_mm=1450"
        )
        self.assertEqual(partial.status_code, 400)
        inverted = self.client.get(
            f"{self.api}/view?layer=station&revision=0"
            "&station_start_mm=1540&station_end_mm=1450"
            "&transverse_start_mm=8&transverse_end_mm=80"
        )
        self.assertEqual(inverted.status_code, 400)

    def test_block_probe_reports_partial_evidence(self):
        from rollscope.probe import point_measurement

        context = AnalysisContext(build_demo("cold-rolled"), AnalysisSettings.defaults())
        spec = RasterSpec(Layer.SPOTS, max_width=120, max_height=300)
        mapping = position_mapping(context, spec)
        self.assertTrue(mapping.block_mode)
        block = mapping.block_at_pixel(0.5, 0.5)
        result = point_measurement(context, spec, 0.5, 0.5)
        self.assertEqual(result["probe_kind"], "block")
        self.assertEqual((result["block_row"], result["block_column"]), block)
        self.assertLessEqual(
            result["station_start_mm"], result["station_end_mm"]
        )
        self.assertLess(
            result["transverse_start_mm"], result["transverse_end_mm"]
        )
        self.assertIn("partial_evidence", result)
        self.assertEqual(result["region"]["station_start_mm"], result["station_start_mm"])
        support = result["support"]
        self.assertEqual(
            support["total_cells"],
            support["native_cells"] + support["blend_cells"] + support["gap_cells"],
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
