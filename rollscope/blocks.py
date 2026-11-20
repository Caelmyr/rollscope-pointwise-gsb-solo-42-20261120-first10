"""Block reduction for station-grid previews.

When a full roll is squeezed into a preview, one display pixel covers many
source cells.  A plain centre sample loses narrow missing lines and small
spots whenever the block centre lands on a normal cell.  The reduction here
keeps a median base tone for the broad bright/dark trend and records the
extreme and minority evidence inside every block so that sub-cell anomalies
can be drawn as a partial mark instead of being averaged away or expanded to
fill the whole display cell.
"""

from dataclasses import dataclass

import numpy as np

from .grid import Support


def _split_edges(start, stop, block_count):
    """Integer boundaries splitting [start, stop) into near-equal blocks.

    The remainder is spread one cell at a time over the first blocks so every
    block is non-empty and ``len(edges) == block_count + 1`` exactly.
    """
    length = stop - start
    base, remainder = divmod(length, block_count)
    edges = [start]
    cursor = start
    for index in range(block_count):
        cursor += base + (1 if index < remainder else 0)
        edges.append(cursor)
    return np.array(edges, dtype=np.int64)


@dataclass(frozen=True)
class BlockGrid:
    """Near-uniform partition of a source row/column range."""

    row_edges: np.ndarray
    column_edges: np.ndarray

    @property
    def block_rows(self):
        return len(self.row_edges) - 1

    @property
    def block_columns(self):
        return len(self.column_edges) - 1

    @property
    def shape(self):
        return (self.block_rows, self.block_columns)

    @property
    def row_count(self):
        return int(self.row_edges[-1] - self.row_edges[0])

    @property
    def column_count(self):
        return int(self.column_edges[-1] - self.column_edges[0])

    @property
    def downsampled(self):
        return self.block_rows < self.row_count or self.block_columns < self.column_count

    @classmethod
    def make(cls, rows, columns, max_height, max_width):
        return cls(
            _split_edges(rows.start, rows.stop, min(rows.stop - rows.start, max_height)),
            _split_edges(columns.start, columns.stop,
                         min(columns.stop - columns.start, max_width)),
        )

    def row_bounds(self, block):
        return int(self.row_edges[block]), int(self.row_edges[block + 1])

    def column_bounds(self, block):
        return int(self.column_edges[block]), int(self.column_edges[block + 1])


def _row_block_index(edges, row_count):
    """Block index per source row, shaped for broadcasting over columns."""
    counts = np.diff(edges)
    return np.repeat(np.arange(len(counts), dtype=np.int64), counts).reshape(row_count, 1)


def _column_block_index(edges, column_count):
    """Block index per source column, shaped for broadcasting over rows."""
    counts = np.diff(edges)
    return np.repeat(np.arange(len(counts), dtype=np.int64), counts).reshape(1, column_count)


def _block_keys(source_shape, block_columns, row_index, column_index):
    rows = np.broadcast_to(row_index, source_shape)
    columns = np.broadcast_to(column_index, source_shape)
    return rows * block_columns + columns


def _count_blocks(mask, keys, shape):
    counts = np.zeros(shape, dtype=np.int64)
    np.add.at(counts.ravel(), keys[mask], 1)
    return counts


def _sorted_finite(values, keys):
    """Group finite values by block and sort each group ascending."""
    finite = np.isfinite(values)
    selected_keys = keys[finite]
    selected_values = values[finite]
    order = np.lexsort((selected_values, selected_keys))
    return selected_keys[order], selected_values[order]


def _block_extremes(values, keys, shape):
    """Finite min/max per block; NaN for blocks without a finite value."""
    minimum = np.full(shape, np.nan, dtype=np.float64)
    maximum = np.full(shape, np.nan, dtype=np.float64)
    block_keys, sorted_values = _sorted_finite(values, keys)
    if block_keys.size:
        unique, first = np.unique(block_keys, return_index=True)
        minimum.flat[unique] = sorted_values[first]
        last = np.r_[first[1:] - 1, len(block_keys) - 1]
        maximum.flat[unique] = sorted_values[last]
    return minimum, maximum


def _block_median(values, keys, shape):
    """Median of the finite values in each block."""
    median = np.full(shape, np.nan, dtype=np.float64)
    block_keys, sorted_values = _sorted_finite(values, keys)
    if not block_keys.size:
        return median
    unique, first, counts = np.unique(block_keys, return_index=True, return_counts=True)
    lower = first + (counts - 1) // 2
    upper = first + counts // 2
    result = 0.5 * (sorted_values[lower] + sorted_values[upper])
    median.flat[unique] = result
    return median


@dataclass(frozen=True)
class BlockFacts:
    """Statistics carried by each display block for station-grid layers."""

    grid: BlockGrid
    median: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    valid_cells: np.ndarray
    gap_cells: np.ndarray
    bad_cells: np.ndarray
    native_cells: np.ndarray
    blend_cells: np.ndarray
    total_cells: np.ndarray
    # Signed-contrast extremes (spots/streaks layers), NaN elsewhere.
    contrast_min: np.ndarray
    contrast_max: np.ndarray
    spot_ids: tuple
    # Number of source cells whose residual crosses the anomaly threshold.
    dark_anomaly_cells: np.ndarray
    bright_anomaly_cells: np.ndarray

    @property
    def shape(self):
        return (self.grid.block_rows, self.grid.block_columns)

    def valid_fraction(self):
        return self.valid_cells / np.maximum(self.total_cells, 1)

    def gap_fraction(self):
        return self.gap_cells / np.maximum(self.total_cells, 1)

    def bad_fraction(self):
        return self.bad_cells / np.maximum(self.total_cells, 1)


def reduce_blocks(context, layer, block_grid):
    """Aggregate every source cell covered by the display blocks."""
    grid = context.grid
    rows = slice(int(block_grid.row_edges[0]), int(block_grid.row_edges[-1]))
    columns = slice(int(block_grid.column_edges[0]), int(block_grid.column_edges[-1]))
    shape = block_grid.shape

    row_index = _row_block_index(block_grid.row_edges, rows.stop - rows.start)
    column_index = _column_block_index(
        block_grid.column_edges, columns.stop - columns.start
    )

    values = grid.values[rows, columns]
    valid = grid.valid[rows, columns]
    support_2d = np.broadcast_to(grid.plan.support[rows, None], values.shape)
    keys = _block_keys(values.shape, block_grid.block_columns, row_index, column_index)

    row_counts = np.diff(block_grid.row_edges)
    column_counts = np.diff(block_grid.column_edges)
    total_cells = (row_counts[:, None] * column_counts[None, :]).astype(np.int64)

    median = _block_median(values, keys, shape)
    minimum, maximum = _block_extremes(values, keys, shape)
    contrast_min = np.full(shape, np.nan)
    contrast_max = np.full(shape, np.nan)
    spot_ids = ()
    dark_anomaly_cells = np.zeros(shape, dtype=np.int64)
    bright_anomaly_cells = np.zeros(shape, dtype=np.int64)

    if layer.value in ("spots", "streaks"):
        report = context.spots if layer.value == "spots" else context.streaks
        contrast = report.residual[rows, columns]
        contrast_min, contrast_max = _block_extremes(contrast, keys, shape)
        threshold = (
            context.settings.spots.contrast
            if layer.value == "spots"
            else context.settings.streaks.contrast
        )
        finite_contrast = np.isfinite(contrast)
        dark_anomaly_cells = _count_blocks(
            finite_contrast & (contrast <= -threshold), keys, shape
        )
        bright_anomaly_cells = _count_blocks(
            finite_contrast & (contrast >= threshold), keys, shape
        )
        if layer.value == "spots":
            labels = report.labels[rows, columns]
            present = np.unique(labels[labels > 0])
            spot_ids = tuple(int(identifier) for identifier in present)

    return BlockFacts(
        grid=block_grid,
        median=median,
        minimum=minimum,
        maximum=maximum,
        valid_cells=_count_blocks(valid, keys, shape),
        gap_cells=_count_blocks(support_2d == Support.GAP, keys, shape),
        bad_cells=_count_blocks(~valid & (support_2d != Support.GAP), keys, shape),
        native_cells=_count_blocks(support_2d == Support.NATIVE, keys, shape),
        blend_cells=_count_blocks(support_2d == Support.BLEND, keys, shape),
        total_cells=total_cells,
        contrast_min=contrast_min,
        contrast_max=contrast_max,
        spot_ids=spot_ids,
        dark_anomaly_cells=dark_anomaly_cells,
        bright_anomaly_cells=bright_anomaly_cells,
    )
