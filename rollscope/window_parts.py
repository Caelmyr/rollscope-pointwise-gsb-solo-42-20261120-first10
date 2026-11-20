import hashlib
import io
from dataclasses import dataclass
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .capture_codec import canonical_json
from .domain import StationWindow, UserInputError
from .evidence import (
    EVIDENCE_FORMAT,
    csv_bytes,
    grid_payload,
    source_payload,
)
from .inspection_sheet import window_sheet
from .raster import Layer, RasterSpec, png_bytes, preview_image
from .records import ReviewWindow
from .window_review import (
    WindowCells,
    review_window,
    row_provenance,
    source_rows,
)
from . import __version__


# Handoff systems reject attachments larger than 2 MiB.
MAX_PART_BYTES = 2 * 1024 * 1024
MAX_WINDOW_PARTS = 24
# Static members (settings, PNGs, CSVs, ZIP overhead) ride on top of the
# measured array payloads. The budget self-calibrates on rebuilds.
INITIAL_OVERHEAD = 220 * 1024
MIN_PAYLOAD_BUDGET = 64 * 1024
CALIBRATION_PASSES = 3
# reflectance (float64-ish), material/valid (int/bool) dominate
# station-cells.npz; raw uint16 samples dominate source-samples.npz.
ROW_WEIGHTS = (8, 4, 1, 2)
SOURCE_LINE_WEIGHT = 2
PARTITION_KIND = "window-part"


@dataclass(frozen=True)
class RowBlock:
    start: int
    stop: int
    splittable: bool
    spot_ids: tuple
    estimated_bytes: int


@dataclass(frozen=True)
class PartPlan:
    index: int
    row_start: int
    row_stop: int
    split_spot_ids: tuple
    estimated_bytes: int


@dataclass(frozen=True)
class BuiltPart:
    index: int
    row_start: int
    row_stop: int
    split_spot_ids: tuple
    bytes: int
    payload: bytes


def spot_row_ranges(context, rows):
    labels = context.spots.labels[rows, :]
    chosen = np.unique(labels[labels > 0])
    ranges = {}
    for label in chosen:
        hit_rows = np.flatnonzero(np.any(labels == int(label), axis=1))
        spot = context.spots.spots[int(label) - 1]
        ranges[int(label)] = (
            int(rows.start + hit_rows[0]),
            int(rows.start + hit_rows[-1]) + 1,
            spot.pixel_count,
        )
    return ranges


def estimate_block_bytes(context, columns, rows):
    cell_columns = columns.stop - columns.start
    row_count = rows.stop - rows.start
    lines = source_rows(context, rows)
    grid_bytes = row_count * cell_columns * sum(ROW_WEIGHTS)
    sample_bytes = len(lines) * cell_columns * SOURCE_LINE_WEIGHT
    return grid_bytes + sample_bytes


def build_row_blocks(context, rows, columns):
    """Group grid rows so every block boundary sits outside every spot.

    Blocks intersecting a spot stay unsplittable during the first packing
    pass, which keeps whole spot observations with one part. Blocks with no
    spot footprint are freely splittable.
    """
    ranges = spot_row_ranges(context, rows)
    cuts = {rows.start, rows.stop}
    for start, stop, _ in ranges.values():
        cuts.add(max(rows.start, start))
        cuts.add(min(rows.stop, stop))
    blocks = []
    for start, stop in zip(sorted(cuts)[:-1], sorted(cuts)[1:]):
        if stop <= start:
            continue
        spot_ids = tuple(
            label
            for label, (spot_start, spot_stop, _) in ranges.items()
            if spot_start < stop and spot_stop > start
        )
        blocks.append(
            RowBlock(
                start,
                stop,
                not spot_ids,
                spot_ids,
                estimate_block_bytes(context, columns, slice(start, stop)),
            )
        )
    return blocks, ranges


def split_oversized_block(block, payload_budget):
    """Last-resort row-level cut for a block larger than one part.

    The pieces keep the original block's spot identifiers so that a spot
    cut between parts can be recognised as one shared observation.
    """
    row_count = block.stop - block.start
    per_row = max(1, block.estimated_bytes // max(1, row_count))
    rows_per_piece = max(1, payload_budget // per_row)
    pieces = []
    start = block.start
    while start < block.stop:
        stop = min(block.stop, start + rows_per_piece)
        share = max(1, (stop - start) * per_row)
        pieces.append(RowBlock(start, stop, True, block.spot_ids, share))
        start = stop
    return pieces


def pack_blocks(blocks, payload_budget):
    groups = []
    current = []
    current_bytes = 0
    for block in blocks:
        if current and current_bytes + block.estimated_bytes > payload_budget:
            # Spot blocks were split upstream only when they alone exceeded
            # a part, so closing the part here never cuts an observation.
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(block)
        current_bytes += block.estimated_bytes
    if current:
        groups.append(current)
    return groups


def make_partition_id(window, rows, columns):
    return hashlib.sha256(
        canonical_json(
            {
                "window_id": window.id,
                "row_start": rows.start,
                "row_stop": rows.stop,
                "column_start": columns.start,
                "column_stop": columns.stop,
            }
        )
    ).hexdigest()[:16]


def plan_parts(context, window, overhead):
    cells = WindowCells.take(context, window.region)
    rows, columns = cells.rows, cells.columns
    payload_budget = MAX_PART_BYTES - overhead
    blocks, ranges = build_row_blocks(context, rows, columns)

    rebuilt = []
    for block in blocks:
        if block.estimated_bytes <= payload_budget:
            rebuilt.append(block)
        else:
            rebuilt.extend(split_oversized_block(block, payload_budget))
    blocks = rebuilt

    groups = pack_blocks(blocks, payload_budget)
    if len(groups) > MAX_WINDOW_PARTS:
        raise UserInputError(
            f"This window needs more than {MAX_WINDOW_PARTS} evidence parts. "
            "Save a shorter review window."
        )

    plans = []
    for index, group in enumerate(groups, 1):
        row_start = group[0].start
        row_stop = group[-1].stop
        split_ids = tuple(
            sorted(
                label
                for label, (start, stop, _) in ranges.items()
                if start < row_stop and stop > row_start
                and (start < row_start or stop > row_stop)
            )
        )
        plans.append(
            PartPlan(
                index,
                row_start,
                row_stop,
                split_ids,
                sum(block.estimated_bytes for block in group),
            )
        )
    partition_id = make_partition_id(window, rows, columns)
    return partition_id, columns, plans, ranges


def part_covers(plan, label, ranges):
    start, stop, _ = ranges[label]
    return start >= plan.row_start and stop <= plan.row_stop


def part_region(context, columns, row_start, row_stop):
    grid = context.grid
    transverse = grid.transverse[columns]
    epsilon = max(1e-9, grid.row_pitch_mm / 4)
    return StationWindow(
        float(grid.stations[row_start]) - epsilon,
        float(grid.stations[row_stop - 1]) + epsilon,
        float(transverse[0]),
        float(transverse[-1]),
    )


def sub_window(window, region):
    return ReviewWindow(window.id, window.label, window.note, window.status, region)


def annotate_split_spots(report, plan, all_plans, ranges):
    for spot in report["spots"]:
        label = spot["spot_id"]
        if label not in plan.split_spot_ids:
            continue
        spot["spot_split"] = True
        spot["spot_part_index"] = plan.index
        spot["spot_observation_parts"] = sorted(
            candidate.index
            for candidate in all_plans
            if label in candidate.split_spot_ids or part_covers(candidate, label, ranges)
        )


def build_part(record, context, window, columns, partition_id, plan, all_plans, ranges):
    region = part_region(context, columns, plan.row_start, plan.row_stop)
    bounded = sub_window(window, region)
    part_cells = WindowCells.take(context, region)
    # Station rounding must never silently move a partition boundary.
    if (part_cells.rows.start, part_cells.rows.stop) != (plan.row_start, plan.row_stop):
        raise UserInputError("The window partition boundary does not align to grid rows.")

    report = review_window(context, bounded)
    annotate_split_spots(report, plan, all_plans, ranges)
    lines = source_rows(context, part_cells.rows)
    image, geometry = preview_image(context, RasterSpec(Layer.SPOTS, window=region))

    members = [
        ("review.json", canonical_json(report)),
        ("settings.json", canonical_json(context.settings.as_dict())),
        ("inspection.png", png_bytes(image)),
        ("station-cells.npz", grid_payload(context, part_cells)),
        ("source-lines.csv", csv_bytes(lines)),
        ("supports.csv", csv_bytes(list(row_provenance(context.grid, part_cells.rows)))),
        ("source-samples.npz", source_payload(context, part_cells, lines)),
        ("spots.csv", csv_bytes(report["spots"])),
        ("gaps.csv", csv_bytes(report["gaps"])),
        ("station-profile.csv", csv_bytes(report["profiles"]["station_bands"])),
        ("transverse-profile.csv", csv_bytes(report["profiles"]["transverse_bands"])),
        ("raster-mapping.json", canonical_json(geometry.as_dict())),
        ("review-sheet.png", png_bytes(window_sheet(record, context, bounded))),
    ]

    binding = {
        "capture_sha256": record.sha256,
        "project_capture_id": record.id,
        "revision": record.state.revision,
        "window_id": window.id,
    }
    partition = {
        "format": EVIDENCE_FORMAT,
        "kind": PARTITION_KIND,
        "partition_id": partition_id,
        "window_id": window.id,
        "window_label": window.label,
        "part_index": plan.index,
        "part_count": len(all_plans),
        "max_part_bytes": MAX_PART_BYTES,
        "row_start": plan.row_start,
        "row_stop": plan.row_stop,
        "station_start_mm": float(context.grid.stations[plan.row_start]),
        "station_end_mm": float(context.grid.stations[plan.row_stop - 1]),
        "transverse_start_mm": float(part_cells.transverse[0]),
        "transverse_end_mm": float(part_cells.transverse[-1]),
        "split_spot_ids": list(plan.split_spot_ids),
        "binding": binding,
    }
    members.append(("partition.json", canonical_json(partition)))
    manifest = {
        "format": EVIDENCE_FORMAT,
        "kind": PARTITION_KIND,
        "engine": __version__,
        "binding": binding,
        "partition": {
            "partition_id": partition_id,
            "part_index": plan.index,
            "part_count": len(all_plans),
        },
        "members": [
            {
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            for name, content in members
        ],
    }
    buffer = io.BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
        archive.writestr("manifest.json", canonical_json(manifest))
        for name, content in members:
            archive.writestr(name, content)
    return buffer.getvalue()


def build_window_partition(record, context, window, overhead):
    partition_id, columns, plans, ranges = plan_parts(context, window, overhead)
    built = []
    for plan in plans:
        payload = build_part(
            record, context, window, columns, partition_id, plan, plans, ranges
        )
        built.append(
            BuiltPart(
                plan.index,
                plan.row_start,
                plan.row_stop,
                plan.split_spot_ids,
                len(payload),
                payload,
            )
        )
    return partition_id, built


def build_partition(record, context, window):
    """Plan and pack parts, tightening the budget if a part overruns 2 MiB."""
    overhead = INITIAL_OVERHEAD
    partition_id, built = build_window_partition(record, context, window, overhead)
    for _ in range(CALIBRATION_PASSES):
        if all(part.bytes <= MAX_PART_BYTES for part in built):
            return partition_id, built
        worst = max(part.bytes / MAX_PART_BYTES for part in built)
        overhead = min(
            MAX_PART_BYTES - MIN_PAYLOAD_BUDGET,
            int(overhead + (worst - 1.0) * MAX_PART_BYTES),
        )
        partition_id, built = build_window_partition(record, context, window, overhead)
    if any(part.bytes > MAX_PART_BYTES for part in built):
        raise UserInputError(
            "This window cannot be packed into 2 MiB evidence parts. "
            "Save a shorter review window."
        )
    return partition_id, built


def partition_plan(record, context, window):
    partition_id, built = build_partition(record, context, window)
    grid = context.grid
    return {
        "partition_id": partition_id,
        "max_part_bytes": MAX_PART_BYTES,
        "parts": [
            {
                "part_index": part.index,
                "part_count": len(built),
                "bytes": part.bytes,
                "station_start_mm": float(grid.stations[part.row_start]),
                "station_end_mm": float(grid.stations[part.row_stop - 1]),
                "split_spot_ids": list(part.split_spot_ids),
            }
            for part in built
        ],
    }
