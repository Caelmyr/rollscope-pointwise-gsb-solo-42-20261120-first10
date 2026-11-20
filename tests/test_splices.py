import unittest
from dataclasses import dataclass

import numpy as np

from rollscope.settings import SpliceSettings
from rollscope.splices import inspect_splices


ROWS = 240
COLS = 120
ROW_PITCH = 0.25
COL_PITCH = 0.5
GAP_START = 110
GAP_STOP = 128


@dataclass(frozen=True)
class FakePlan:
    observed: np.ndarray


@dataclass(frozen=True)
class FakeGrid:
    values: np.ndarray
    valid: np.ndarray
    transverse: np.ndarray
    stations: np.ndarray
    plan: FakePlan
    shape: tuple

    @property
    def row_pitch_mm(self):
        return ROW_PITCH

    @property
    def column_pitch_mm(self):
        return COL_PITCH


@dataclass(frozen=True)
class FakeEdges:
    material: np.ndarray

    def inner_mask(self, margin_cols):
        if margin_cols <= 0:
            return self.material.copy()
        result = self.material.copy()
        result[:, :margin_cols] = False
        result[:, self.material.shape[1] - margin_cols :] = False
        return result


def make_grid(values, observed, valid):
    stations = np.arange(ROWS, dtype=np.float64) * ROW_PITCH
    transverse = np.arange(COLS, dtype=np.float64) * COL_PITCH
    return FakeGrid(
        values.astype(np.float32),
        valid,
        transverse,
        stations,
        FakePlan(observed),
        (ROWS, COLS),
    )


def paint_diagonal(values, valid, observed, start_row, start_col, slope, sign=1,
                   amplitude=0.22, half_width=2.0, stop_row=None):
    """Draw a straight bright (or dark) diagonal line on a flat field."""
    stop_row = ROWS if stop_row is None else stop_row
    rows = np.arange(ROWS)[:, None]
    cols = np.arange(COLS)[None, :]
    centre = start_row + slope * (cols - start_col)
    line = amplitude * np.exp(-0.5 * ((rows - centre) / half_width) ** 2)
    if sign < 0:
        line = -line
    values += np.where(observed[:, None] & valid, line, 0.0)


def base_scenery(gap=True):
    rng = np.random.default_rng(7)
    values = rng.normal(0.5, 0.004, size=(ROWS, COLS)).astype(np.float32)
    valid = np.ones((ROWS, COLS), dtype=bool)
    observed = np.ones(ROWS, dtype=bool)
    material = np.ones((ROWS, COLS), dtype=bool)
    if gap:
        observed[GAP_START:GAP_STOP] = False
        valid[GAP_START:GAP_STOP, :] = False
        material[GAP_START:GAP_STOP, :] = False
        values[GAP_START:GAP_STOP, :] = np.nan
    return values, valid, observed, material


class DiagonalJoinTracking(unittest.TestCase):
    def run_inspection(self, values, observed, valid, material, **overrides):
        grid = make_grid(values, observed, valid)
        edges = FakeEdges(material)
        settings = SpliceSettings(**overrides)
        return inspect_splices(grid, edges, settings), grid

    def test_diagonal_join_tracks_across_encoder_gap_as_one_track(self):
        values, valid, observed, material = base_scenery(gap=True)
        # A gentle slope keeps measured evidence on both sides of the gap while
        # the line's middle crosses only unsampled rows. Start and end columns
        # stay inside the strip so neither fragment is clipped by the margin.
        slope = 0.7
        paint_diagonal(values, valid, observed, 84, 28, slope)
        # 0.7 rows/column = 1.4 mm/mm here, so widen the accepted slope for
        # this deliberately steep synthetic line.
        report, grid = self.run_inspection(
            values, observed, valid, material, max_slope=2.0
        )

        diagonal = [s for s in report.splices if abs(s.slope) > 0.05]
        self.assertGreaterEqual(len(diagonal), 2)
        self.assertEqual(len({s.track_id for s in diagonal}), 1)
        self.assertEqual(
            [s.link for s in diagonal].count("continued"), len(diagonal) - 1
        )
        gap_links = [link for link in report.links if link.gap_kind == "unsupported"]
        self.assertEqual(len(gap_links), 1)
        self.assertEqual(gap_links[0].status, "continued")
        self.assertGreater(gap_links[0].gap_mm, 0)
        # The inferred span covers the unsampled station interval.
        self.assertLessEqual(
            gap_links[0].station_before_mm, grid.stations[GAP_START - 1]
        )
        self.assertGreaterEqual(
            gap_links[0].station_after_mm, grid.stations[GAP_STOP]
        )

    def test_measured_blank_stretch_is_not_bridged(self):
        values, valid, observed, material = base_scenery(gap=False)
        slope = 0.3
        # Two collinear fragments with a fully observed, contrast-free interval.
        paint_diagonal(values, valid, observed, 40, 20, slope, stop_row=70)
        paint_diagonal(values, valid, observed, 40, 20, slope)
        # Remove the line from a long measured interval so no evidence exists.
        values[88:112, :] = 0.5
        report, _ = self.run_inspection(values, observed, valid, material)
        for link in report.links:
            self.assertNotEqual(link.gap_kind, "observed")

    def test_two_close_parallel_lines_remain_separate_tracks(self):
        values, valid, observed, material = base_scenery(gap=False)
        # Same slope but offset in both station and transverse position, with
        # no shared evidence cells, so neither merge rule can join them.
        paint_diagonal(values, valid, observed, 40, 14, 0.2)
        paint_diagonal(values, valid, observed, 96, 14, 0.2)
        report, _ = self.run_inspection(values, observed, valid, material)
        tracks = {s.track_id for s in report.splices if abs(s.slope) > 0.05}
        self.assertGreaterEqual(len(tracks), 2)

    def test_horizontal_join_across_one_station_still_detected(self):
        values, valid, observed, material = base_scenery(gap=False)
        row = 130
        values[row, 10:110] += 0.3
        report, _ = self.run_inspection(values, observed, valid, material)
        horizontal = [s for s in report.splices if abs(s.slope) <= 0.05]
        self.assertTrue(horizontal)
        self.assertTrue(any(s.coverage >= 0.65 for s in horizontal))

    def test_bright_and_dark_polarity_are_kept_distinct(self):
        values, valid, observed, material = base_scenery(gap=False)
        paint_diagonal(values, valid, observed, 45, 18, 0.2, sign=1)
        paint_diagonal(values, valid, observed, 52, 18, 0.2, sign=-1)
        report, _ = self.run_inspection(values, observed, valid, material)
        polarities = {s.polarity for s in report.splices}
        self.assertIn("bright", polarities)
        self.assertIn("dark", polarities)

    def test_ambiguous_match_is_flagged_not_silently_joined(self):
        from rollscope.splices import EvidenceRun, FittedSegment, link_segments

        class TinyGrid:
            row_pitch_mm = ROW_PITCH
            column_pitch_mm = COL_PITCH

        def segment(points, sign=1):
            runs = tuple(
                EvidenceRun(row, col - 3, col + 3, sign, 7, 0.2)
                for row, col in points
            )
            centres = np.array([0.5 * (run.first_col + run.last_col) for run in runs])
            rows = np.array([run.row for run in runs])
            slope, intercept = np.polyfit(centres, rows, 1)
            return FittedSegment(sign, runs, float(intercept), float(slope), 0.5)

        observed = np.ones(ROWS, dtype=bool)
        observed[GAP_START:GAP_STOP] = False
        before = segment([(80 + i, 30 + i) for i in range(16)])
        # Two post-gap fragments, both within the tracking tolerance of the
        # extrapolated pre-gap line, so the match cannot be decided uniquely.
        after_a = segment([(130 + i, 80 + i) for i in range(16)])
        after_b = segment([(130 + i, 76 + i) for i in range(16)])
        settings = SpliceSettings(track_tolerance_mm=8.0, max_gap_mm=40.0)
        links, tracks = link_segments(
            [before, after_a, after_b], observed, settings, TinyGrid()
        )
        self.assertTrue(any(link[4] == "ambiguous" for link in links))
        # The ambiguous continuation is an unsupported stretch, and the two
        # candidates are not collapsed into a single track.
        ambiguous = [link for link in links if link[4] == "ambiguous"]
        self.assertTrue(all(link[3] == "unsupported" for link in ambiguous))
        candidate_tracks = {tracks[id(after_a)], tracks[id(after_b)]}
        self.assertEqual(len(candidate_tracks), 2)


if __name__ == "__main__":
    unittest.main()
