import io
import json
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from rollscope.analysis import SECTION_TITLES, AnalysisContext, AnalysisPresenter
from rollscope.calibration import ReferenceBasis, calibrate, endpoint_profile
from rollscope.capture_codec import capture_bytes, capture_hash, read_capture
from rollscope.demo import PROFILES, build_demo
from rollscope.domain import Capture, CaptureMetadata, Geometry, UserInputError
from rollscope.grid import Support
from rollscope.http import create_app
from rollscope.lineage import detailed_lineage, station_lineage
from rollscope.ruler import Ruler, evaluate_ruler
from rollscope.settings import AnalysisSettings


def single_reference_capture(capture):
    """Rebuild a capture using only its start references (legacy format)."""
    return Capture.build(
        capture.raw,
        capture.ticks,
        capture.seconds,
        capture.dark,
        capture.white,
        capture.metadata,
    )


def synthetic_capture(rows=120, columns=48, paired=True):
    rng = np.random.default_rng(71)
    ticks = np.cumsum(np.full(rows, 2, dtype=np.int64))
    seconds = np.cumsum(np.full(rows, 0.01, dtype=np.float64))
    geometry = Geometry(0.2, 0.5, 0.0, 0.0)
    metadata = CaptureMetadata("Probe coil", "Test material", "Synthetic", True, geometry)
    dark = np.rint(700 + rng.normal(0, 8, size=(4, columns))).astype(np.uint16)
    white = np.rint(30000 + rng.normal(0, 20, size=(5, columns))).astype(np.uint16)
    raw = np.rint(15000 + rng.normal(0, 120, size=(rows, columns))).clip(0, 65535).astype(np.uint16)
    if not paired:
        return Capture.build(raw, ticks, seconds, dark, white, metadata)
    dark_end = np.rint(730 + rng.normal(0, 8, size=(4, columns))).astype(np.uint16)
    white_end = np.rint(31200 + rng.normal(0, 20, size=(5, columns))).astype(np.uint16)
    # Column 10 saturates at the end event only.
    white_end[:, 10] = 65535
    reference_seconds = np.array([seconds[0] - 0.5, seconds[-1] + 0.5])
    return Capture.build(
        raw, ticks, seconds, dark, white, metadata,
        dark_end=dark_end, white_end=white_end, reference_seconds=reference_seconds,
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
        self.assertEqual(context.calibrated.start.summary()["rejected_columns"], 3)
        # Column 291 fails at both reference events, so it never has a basis.
        self.assertTrue(np.all(context.calibrated.basis[:, 291] == ReferenceBasis.UNKNOWN.value))
        self.assertTrue(np.all(~context.grid.valid[:, 291]))
        self.assertTrue(np.isnan(context.grid.values[:, 291]).all())
        self.assertEqual(len(context.coverage.gaps), 2)

    def test_paired_references_transition_and_trust_reach(self):
        context = self.context
        calibrated = context.calibrated
        self.assertEqual(calibrated.mode, "paired")
        self.assertEqual(len(calibrated.reference_seconds), 2)
        # A healthy column blends both references across the whole scan.
        column = 100
        self.assertTrue(np.all(calibrated.basis[:, column] == ReferenceBasis.BLENDED.value))
        first_weight, first_end = calibrated.reference_weights(0, column)
        last_weight, last_end = calibrated.reference_weights(calibrated.basis.shape[0] - 1, column)
        self.assertAlmostEqual(first_weight, 1.0)
        self.assertAlmostEqual(first_end, 0.0)
        self.assertAlmostEqual(last_weight, 0.0)
        self.assertAlmostEqual(last_end, 1.0)
        # Column 302 fails only at the end: the head uses the good start,
        # the tail beyond the trust reach is explicitly unknown.
        basis = calibrated.basis[:, 302]
        self.assertEqual(ReferenceBasis(basis[0]), ReferenceBasis.START)
        middle = len(basis) // 2
        self.assertEqual(ReferenceBasis(basis[middle]), ReferenceBasis.UNKNOWN)
        self.assertFalse(calibrated.valid[middle, 302])
        self.assertTrue(np.isnan(calibrated.values[middle, 302]))
        last_known = np.flatnonzero(basis[:middle] != ReferenceBasis.UNKNOWN.value)[-1]
        self.assertLessEqual(calibrated.transition[last_known], 0.25 + 1e-6)
        # Column 22 fails only at the start: the tail is end-supported.
        self.assertEqual(ReferenceBasis(calibrated.basis[-1, 22]), ReferenceBasis.END)
        # A column rejected at both events is unknown on every line.
        self.assertTrue(np.all(calibrated.basis[:, 64] == ReferenceBasis.UNKNOWN.value))

    def test_probe_reports_reference_basis_and_source_lines(self):
        context = self.context
        column = 100
        row = context.capture.rows // 2
        lineage = detailed_lineage(context, row, column, "flat")
        entries = lineage["segments"][0]["rows"]
        self.assertTrue(all("reference_basis" in item for item in entries))
        selected = next(item for item in entries if item["selected_source"])
        self.assertEqual(selected["reference_basis"], "blended")
        self.assertGreater(selected["start_reference_weight"], 0.0)
        self.assertGreater(selected["end_reference_weight"], 0.0)
        weak_middle = detailed_lineage(context, row, 302, "flat")["segments"][0]["rows"]
        weak_row = next(item for item in weak_middle if item["selected_source"])
        self.assertEqual(weak_row["reference_basis"], "unknown")
        self.assertEqual(weak_row["end_reference_weight"], 0.0)
        self.assertEqual(weak_row["start_reference_weight"], 0.0)

    def test_flat_presentation_exposes_both_reference_events(self):
        result = AnalysisPresenter(self.context, 900).output("flat")
        summary = result["summary"]
        self.assertEqual(summary["reference_calibration"]["mode"], "paired")
        self.assertEqual(summary["reference_events"], 2)
        self.assertIn("end", summary["reference_calibration"])
        titles = [chart["title"] for chart in result["charts"]]
        self.assertIn("Start-to-end reference drift", titles)
        self.assertIn("Reference basis along the scan", titles)
        flagged = next(table for table in result["tables"] if table["key"] == "flagged_columns")
        self.assertTrue(any(row["column"] == 302 for row in flagged["rows"]))
        dark_passes = next(table for table in result["tables"] if table["key"] == "dark_passes")
        events = {row["event"] for row in dark_passes["rows"]}
        self.assertEqual(events, {"start", "end"})

    def test_paired_bundle_roundtrips_while_legacy_bundle_still_opens(self):
        paired = self.capture
        restored = read_capture(capture_bytes(paired))
        self.assertTrue(restored.paired_references)
        self.assertEqual(capture_hash(paired), capture_hash(restored))
        np.testing.assert_array_equal(restored.dark_end, paired.dark_end)
        np.testing.assert_array_equal(restored.white_end, paired.white_end)
        np.testing.assert_allclose(restored.reference_seconds, paired.reference_seconds)
        legacy = single_reference_capture(paired)
        reopened = read_capture(capture_bytes(legacy))
        self.assertFalse(reopened.paired_references)
        legacy_context = AnalysisContext(reopened, AnalysisSettings.defaults())
        self.assertEqual(legacy_context.calibrated.mode, "single")
        # Legacy single-set data keeps every usable column valid wherever the
        # raw sample itself is not saturated.
        good_columns = legacy_context.calibrated.start.usable
        self.assertTrue(
            np.all(
                legacy_context.calibrated.valid[:, good_columns]
                == ~legacy_context.calibrated.clipped[:, good_columns]
            )
        )
        self.assertTrue(
            np.all(legacy_context.calibrated.basis[:, good_columns] == ReferenceBasis.START.value)
        )

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


class PairedReferenceRules(unittest.TestCase):
    def test_trust_reach_setting_controls_unknown_region(self):
        capture = synthetic_capture()
        narrow = AnalysisSettings.parse({"calibration": {"reference_trust_reach_fraction": 0.1}})
        wide = AnalysisSettings.parse({"calibration": {"reference_trust_reach_fraction": 0.45}})
        narrow_image = calibrate(capture, narrow.calibration)
        wide_image = calibrate(capture, wide.calibration)
        # Column 10 is end-rejected but start-valid: only the head is supported.
        unknown_rows_narrow = np.count_nonzero(
            narrow_image.basis[:, 10] == ReferenceBasis.UNKNOWN.value
        )
        unknown_rows_wide = np.count_nonzero(
            wide_image.basis[:, 10] == ReferenceBasis.UNKNOWN.value
        )
        self.assertGreater(unknown_rows_narrow, unknown_rows_wide)
        self.assertEqual(
            ReferenceBasis(narrow_image.basis[0, 10]), ReferenceBasis.START
        )

    def test_zero_reach_requires_both_reference_events(self):
        capture = synthetic_capture()
        settings = AnalysisSettings.parse({"calibration": {"reference_trust_reach_fraction": 0.0}})
        image = calibrate(capture, settings.calibration)
        # With no single-side reach, a one-sided column is unknown beyond its
        # own exact endpoint line (fraction == 0).
        self.assertEqual(ReferenceBasis(image.basis[1, 10]), ReferenceBasis.UNKNOWN)
        healthy = np.setdiff1d(np.arange(capture.columns), [10])
        self.assertTrue(
            np.all(image.basis[:, healthy] == ReferenceBasis.BLENDED.value)
        )

    def test_mismatched_paired_references_are_rejected(self):
        capture = synthetic_capture()
        with self.assertRaises(UserInputError):
            Capture.build(
                capture.raw, capture.ticks, capture.seconds,
                capture.dark, capture.white, capture.metadata,
                dark_end=capture.dark_end,
            )
        with self.assertRaises(UserInputError):
            Capture.build(
                capture.raw, capture.ticks, capture.seconds,
                capture.dark, capture.white, capture.metadata,
                white_end=capture.white_end,
            )

    def test_reference_clock_must_have_one_value_per_event(self):
        capture = synthetic_capture()
        with self.assertRaises(UserInputError):
            Capture.build(
                capture.raw, capture.ticks, capture.seconds,
                capture.dark, capture.white, capture.metadata,
                dark_end=capture.dark_end, white_end=capture.white_end,
                reference_seconds=np.array([1.0]),
            )
        with self.assertRaises(UserInputError):
            Capture.build(
                capture.raw, capture.ticks, capture.seconds,
                capture.dark, capture.white, capture.metadata,
                reference_seconds=np.array([1.0, 2.0]),
            )

    def test_both_demo_profiles_are_paired_and_legacy_profile_remains_single(self):
        for name in PROFILES:
            self.assertTrue(build_demo(name).paired_references)
        capture = build_demo("coated")
        legacy = single_reference_capture(capture)
        self.assertFalse(legacy.paired_references)
        self.assertEqual(legacy.overview()["reference_events"], 1)


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

    def test_flat_probe_and_trust_reach_setting_round_trip(self):
        probe = self.client.get(
            f"{self.api}/probe?layer=flat&revision=0&x=0.5&y=0.5"
        )
        self.assertEqual(probe.status_code, 200)
        body = probe.json
        self.assertEqual(body["reference_basis"], "blended")
        self.assertGreater(body["start_reference_weight"], 0.0)
        self.assertGreater(body["end_reference_weight"], 0.0)
        self.assertEqual(len(body["lineage"]["segments"][0]["rows"]), 9)
        self.assertIn("reference_basis", body["lineage"]["segments"][0]["rows"][0])

        applied = self.client.post(f"{self.api}/settings", json={
            "revision": 0,
            "section": "calibration",
            "values": {"reference_trust_reach_fraction": 0.4},
        })
        self.assertEqual(applied.status_code, 200, applied.json)
        state = applied.json["state"]
        self.assertEqual(
            state["settings"]["calibration"]["reference_trust_reach_fraction"], 0.4
        )
        flat = self.client.get(f"{self.api}/analysis/flat?revision=1").json
        self.assertEqual(
            flat["summary"]["reference_calibration"]["trust_reach_fraction"], 0.4
        )
        rejected = self.client.post(f"{self.api}/settings", json={
            "revision": 1,
            "section": "calibration",
            "values": {"reference_trust_reach_fraction": 2.0},
        })
        self.assertEqual(rejected.status_code, 400)

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
