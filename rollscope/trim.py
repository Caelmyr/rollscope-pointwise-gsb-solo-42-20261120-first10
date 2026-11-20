from dataclasses import dataclass
from enum import Enum

import numpy as np

from .domain import UserInputError, exact_keys, expect_object, finite_number
from .grid import Support


MAX_REMOVED_SPOTS = 3
MAX_CUT_REGIONS = 1
CUT_LIMIT = 2


class TrimObjective(str, Enum):
    LONGEST = "longest_segment"
    RETAIN = "max_retained"


OBJECTIVE_TITLES = {
    TrimObjective.LONGEST: "Longest continuous good material",
    TrimObjective.RETAIN: "Maximum retained material",
}


def conflict(code, message, boundary=None):
    entry = {"code": code, "message": message}
    if boundary is not None:
        entry["boundary"] = boundary
    return entry


def optional_station(value, label):
    if value is None:
        return None
    return finite_number(value, label, -1e8, 1e8)


@dataclass(frozen=True)
class TrimRequest:
    margin_mm: float
    minimum_length_mm: float
    retain_start_mm: float | None
    retain_end_mm: float | None
    spot_ids: tuple

    @classmethod
    def parse(cls, payload):
        data = expect_object(payload, "Trim request")
        exact_keys(
            data,
            ("margin_mm", "minimum_length_mm", "retain_start_mm", "retain_end_mm", "spot_ids"),
            "Trim request",
        )
        margin = finite_number(data["margin_mm"], "Trim margin", 0.0, 5000.0)
        minimum = finite_number(data["minimum_length_mm"], "Minimum shippable length", 0.0, 100000.0)
        retain_start = optional_station(data["retain_start_mm"], "Retained sample start")
        retain_end = optional_station(data["retain_end_mm"], "Retained sample end")
        if (retain_start is None) != (retain_end is None):
            raise UserInputError("Provide both retained sample bounds or leave both empty.")
        if retain_start is not None and retain_start >= retain_end:
            raise UserInputError("The retained sample must have positive length.")
        ids = data["spot_ids"]
        if not isinstance(ids, list) or not all(type(item) is int for item in ids):
            raise UserInputError("Spot selections must be whole-number spot identifiers.")
        if len(ids) > MAX_REMOVED_SPOTS:
            raise UserInputError(
                f"At most {MAX_REMOVED_SPOTS} spots can be removed in this downtime window."
            )
        if len(set(ids)) != len(ids):
            raise UserInputError("Each spot can only be selected once.")
        return cls(margin, minimum, retain_start, retain_end, tuple(ids))


@dataclass(frozen=True)
class Cut:
    station_mm: float
    row: int
    support: str
    valid_cells: int

    def as_dict(self):
        return {
            "station_mm": self.station_mm,
            "row": self.row,
            "coverage": self.support,
            "valid_cells": self.valid_cells,
        }


@dataclass(frozen=True)
class CutRegion:
    start_mm: float
    end_mm: float
    start_row: int
    end_row: int
    length_mm: float
    spot_ids: tuple
    start_cut: Cut
    end_cut: Cut

    def as_dict(self):
        return {
            "start_mm": self.start_mm,
            "end_mm": self.end_mm,
            "start_row": self.start_row,
            "end_row": self.end_row,
            "length_mm": self.length_mm,
            "spot_ids": list(self.spot_ids),
            "start_cut": self.start_cut.as_dict(),
            "end_cut": self.end_cut.as_dict(),
        }


@dataclass(frozen=True)
class RemnantSegment:
    index: int
    start_mm: float
    end_mm: float
    length_mm: float
    shippable: bool
    contains_retain: bool

    def as_dict(self):
        return {
            "segment": self.index,
            "start_mm": self.start_mm,
            "end_mm": self.end_mm,
            "length_mm": self.length_mm,
            "shippable": self.shippable,
            "contains_retain": self.contains_retain,
        }


@dataclass(frozen=True)
class SelectedSpot:
    spot_id: int
    footprint_start_mm: float
    footprint_end_mm: float
    isolation_start_mm: float
    isolation_end_mm: float
    partial: bool
    polarity: str

    def as_dict(self):
        return {
            "spot_id": self.spot_id,
            "footprint_start_mm": self.footprint_start_mm,
            "footprint_end_mm": self.footprint_end_mm,
            "isolation_start_mm": self.isolation_start_mm,
            "isolation_end_mm": self.isolation_end_mm,
            "partial": self.partial,
            "polarity": self.polarity,
        }


@dataclass(frozen=True)
class TrimPlan:
    objective: str
    feasible: bool
    cut_count: int
    removed_length_mm: float
    retained_length_mm: float
    longest_segment_mm: float
    shippable_length_mm: float
    shippable_segment_count: int
    selected_spots: tuple
    cut_regions: tuple
    segments: tuple
    retain: object
    conflicts: tuple
    warnings: tuple
    total_length_mm: float

    def as_dict(self):
        return {
            "objective": self.objective,
            "feasible": self.feasible,
            "cut_count": self.cut_count,
            "removed_length_mm": self.removed_length_mm,
            "retained_length_mm": self.retained_length_mm,
            "longest_segment_mm": self.longest_segment_mm,
            "shippable_length_mm": self.shippable_length_mm,
            "shippable_segment_count": self.shippable_segment_count,
            "selected_spots": [item.as_dict() for item in self.selected_spots],
            "cut_regions": [item.as_dict() for item in self.cut_regions],
            "segments": [item.as_dict() for item in self.segments],
            "retain": self.retain,
            "conflicts": list(self.conflicts),
            "warnings": list(self.warnings),
            "total_length_mm": self.total_length_mm,
        }


class TrimPlanner:
    def __init__(self, context):
        self.context = context
        grid = context.grid
        self.grid = grid
        self.stations = grid.stations
        self.pitch = float(grid.row_pitch_mm)
        observed = grid.plan.support != Support.GAP
        # A cut needs a real sampled station with at least one valid sampled cell.
        self.cut_rows = np.flatnonzero(observed & np.any(grid.valid, axis=1))
        self.observed = observed
        self.roll_start = float(self.stations[0])
        self.roll_end = float(self.stations[-1])
        self.total_length = float(self.roll_end - self.roll_start)
        self.spots = {spot.id: spot for spot in context.spots.spots}

    def make_cut(self, row):
        return Cut(
            float(self.stations[row]),
            int(row),
            Support(int(self.grid.plan.support[row])).name.lower(),
            int(np.count_nonzero(self.grid.valid[row])),
        )

    def snap_outward(self, station, side):
        """Nearest sampled row with a valid cell at or beyond the requested station."""
        if side == "left":
            pos = int(np.searchsorted(self.stations, station, side="left"))
            pos = max(0, min(pos, len(self.stations) - 1))
            candidates = self.cut_rows[self.cut_rows <= pos]
            return int(candidates[-1]) if candidates.size else None
        pos = int(np.searchsorted(self.stations, station, side="right")) - 1
        pos = max(0, min(pos, len(self.stations) - 1))
        candidates = self.cut_rows[self.cut_rows >= pos]
        return int(candidates[0]) if candidates.size else None

    def select_spots(self, spot_ids, margin):
        chosen = []
        missing = []
        for spot_id in spot_ids:
            spot = self.spots.get(spot_id)
            if spot is None:
                missing.append(spot_id)
                continue
            chosen.append(
                SelectedSpot(
                    spot.id,
                    float(spot.from_mm),
                    float(spot.to_mm),
                    float(spot.from_mm) - margin,
                    float(spot.to_mm) + margin,
                    bool(spot.partial),
                    spot.polarity,
                )
            )
        return chosen, missing

    def merged_intervals(self, selected):
        ordered = sorted(selected, key=lambda item: item.isolation_start_mm)
        merged = []
        for spot in ordered:
            if not merged or spot.isolation_start_mm > merged[-1][1]:
                merged.append(
                    [spot.isolation_start_mm, spot.isolation_end_mm, [spot.spot_id]]
                )
            else:
                merged[-1][1] = max(merged[-1][1], spot.isolation_end_mm)
                merged[-1][2].append(spot.spot_id)
        return merged

    def build_regions(self, selected):
        conflicts = []
        regions = []
        merged = self.merged_intervals(selected)
        if len(merged) > MAX_CUT_REGIONS:
            conflicts.append(
                conflict(
                    "cut_window_exceeded",
                    f"Two cuts can isolate one contiguous interval, but the selected spots form "
                    f"{len(merged)} separate isolation intervals, so {len(merged) - 1} boundary cannot be confirmed.",
                )
            )
            return regions, conflicts
        for index, (start, end, ids) in enumerate(merged, 1):
            left_row = self.snap_outward(start, "left")
            right_row = self.snap_outward(end, "right")
            boundary = {
                "region": index,
                "requested_start_mm": start,
                "requested_end_mm": end,
            }
            if left_row is None or right_row is None:
                conflicts.append(
                    conflict(
                        "no_sampled_cut",
                        "No sampled station with a valid cell supports the requested cut, so this "
                        "boundary cannot be confirmed.",
                        boundary,
                    )
                )
                continue
            if left_row >= right_row:
                conflicts.append(
                    conflict(
                        "unsupported_isolation",
                        "The isolation interval straddles an encoder gap with no sampled cut "
                        "station on both sides, so its boundary cannot be confirmed.",
                        boundary,
                    )
                )
                continue
            regions.append(
                CutRegion(
                    float(self.stations[left_row]),
                    float(self.stations[right_row]),
                    left_row,
                    right_row,
                    float(self.stations[right_row] - self.stations[left_row]),
                    tuple(sorted(ids)),
                    self.make_cut(left_row),
                    self.make_cut(right_row),
                )
            )
        return regions, conflicts

    def build_segments(self, regions):
        bounds = [self.roll_start]
        for region in regions:
            bounds.extend((region.start_mm, region.end_mm))
        bounds.append(self.roll_end)
        return tuple(
            RemnantSegment(index, float(start), float(end), float(end - start), False, False)
            for index, (start, end) in enumerate(zip(bounds[::2], bounds[1::2]), 1)
        )

    def evaluate_retain(self, segments, request, regions):
        conflicts = []
        if request.retain_start_mm is None:
            return segments, conflicts, None
        retain = {
            "start_mm": request.retain_start_mm,
            "end_mm": request.retain_end_mm,
        }
        if request.retain_start_mm < self.roll_start or request.retain_end_mm > self.roll_end:
            conflicts.append(
                conflict(
                    "retain_outside_roll",
                    "The retained sample extends beyond the captured roll bounds, so its edges cannot be confirmed.",
                    retain,
                )
            )
        for region in regions:
            if request.retain_end_mm > region.start_mm and request.retain_start_mm < region.end_mm:
                conflicts.append(
                    conflict(
                        "retain_inside_cutout",
                        "The retained sample overlaps material that would be cut away, so this boundary cannot be confirmed.",
                        retain,
                    )
                )
        updated = tuple(
            RemnantSegment(
                segment.index,
                segment.start_mm,
                segment.end_mm,
                segment.length_mm,
                segment.shippable,
                request.retain_start_mm >= segment.start_mm - 1e-9
                and request.retain_end_mm <= segment.end_mm + 1e-9,
            )
            for segment in segments
        )
        if not any(item.contains_retain for item in updated) and not conflicts:
            conflicts.append(
                conflict(
                    "retain_spans_cut",
                    "The retained sample spans a planned cut line instead of sitting inside one remnant.",
                    retain,
                )
            )
        return updated, conflicts, retain

    def mark_shippable(self, segments, minimum):
        return tuple(
            RemnantSegment(
                segment.index,
                segment.start_mm,
                segment.end_mm,
                segment.length_mm,
                segment.length_mm + 1e-9 >= minimum,
                segment.contains_retain,
            )
            for segment in segments
        )

    def warnings_for(self, selected, regions):
        warnings = []
        for spot in selected:
            if spot.partial:
                warnings.append(
                    f"Spot {spot.spot_id} touches unsupported evidence; its footprint extent is not fully observed."
                )
        for region in regions:
            gap_rows = int(
                np.count_nonzero(
                    self.grid.plan.support[region.start_row : region.end_row + 1] == Support.GAP
                )
            )
            if gap_rows:
                warnings.append(
                    f"The cut-out interval {region.start_mm:.2f}–{region.end_mm:.2f} mm contains "
                    f"{gap_rows} unsampled station row(s)."
                )
        return tuple(warnings)

    def assemble(self, objective, request, selected, region_conflicts):
        regions, snap_conflicts = self.build_regions(selected)
        conflicts = list(region_conflicts) + snap_conflicts
        segments = self.build_segments(regions)
        segments, retain_conflicts, retain = self.evaluate_retain(segments, request, regions)
        conflicts.extend(retain_conflicts)
        segments = self.mark_shippable(segments, request.minimum_length_mm)
        removed = sum(region.length_mm for region in regions)
        retained = self.total_length - removed
        shippable = [segment for segment in segments if segment.shippable]
        longest = max((segment.length_mm for segment in segments), default=0.0)
        feasible = not conflicts and len(regions) <= MAX_CUT_REGIONS
        return TrimPlan(
            objective.value,
            feasible,
            cut_count=sum(2 for _ in regions),
            removed_length_mm=float(removed),
            retained_length_mm=float(retained),
            longest_segment_mm=float(longest),
            shippable_length_mm=float(sum(item.length_mm for item in shippable)),
            shippable_segment_count=len(shippable),
            selected_spots=tuple(sorted(selected, key=lambda item: item.spot_id)),
            cut_regions=tuple(regions),
            segments=segments,
            retain=retain,
            conflicts=tuple(conflicts),
            warnings=self.warnings_for(selected, regions),
            total_length_mm=self.total_length,
        )

    def empty_plan(self, objective, request, conflicts=()):
        shippable = self.total_length + 1e-9 >= request.minimum_length_mm
        return TrimPlan(
            objective.value,
            feasible=not conflicts,
            cut_count=0,
            removed_length_mm=0.0,
            retained_length_mm=self.total_length,
            longest_segment_mm=self.total_length,
            shippable_length_mm=self.total_length if shippable else 0.0,
            shippable_segment_count=1 if shippable else 0,
            selected_spots=(),
            cut_regions=(),
            segments=(
                RemnantSegment(
                    1, self.roll_start, self.roll_end, self.total_length, shippable, False
                ),
            ),
            retain=(
                None
                if request.retain_start_mm is None
                else {
                    "start_mm": request.retain_start_mm,
                    "end_mm": request.retain_end_mm,
                }
            ),
            conflicts=tuple(conflicts),
            warnings=(),
            total_length_mm=self.total_length,
        )

    def plan(self, objective, request, pre_conflicts=()):
        if not isinstance(objective, TrimObjective):
            raise UserInputError("Choose a valid trim objective.")
        selected, missing = self.select_spots(request.spot_ids, request.margin_mm)
        conflicts = list(pre_conflicts) + [
            conflict(
                "unknown_spot",
                f"Spot {spot_id} is not present in the current surface-spot result.",
                {"spot_id": spot_id},
            )
            for spot_id in missing
        ]
        if not selected:
            return self.empty_plan(objective, request, conflicts)
        return self.assemble(objective, request, selected, conflicts)

    def isolation_clusters(self, request):
        """Connected components of spot isolation intervals along the roll."""
        all_spots, _ = self.select_spots(sorted(self.spots), request.margin_mm)
        clusters = []
        for spot in sorted(all_spots, key=lambda item: item.isolation_start_mm):
            if not clusters or spot.isolation_start_mm > clusters[-1][1]:
                clusters.append(
                    [spot.isolation_start_mm, spot.isolation_end_mm, [spot]]
                )
            else:
                clusters[-1][1] = max(clusters[-1][1], spot.isolation_end_mm)
                clusters[-1][2].append(spot)
        return clusters

    def feasible_subsets(self, request):
        """Every feasible spot subset removing up to three spots under the two-cut limit.

        A subset is feasible only when its isolation intervals merge into one connected
        interval, so the subsets can be enumerated per connected isolation cluster.
        Clusters larger than the removal limit still contribute their smaller sub-subsets.
        """
        clusters = self.isolation_clusters(request)
        subsets = [()]
        for start, end, spots in clusters:
            members = [spot.spot_id for spot in spots]
            size = len(members)
            for first in range(size):
                subsets.append((members[first],))
                for second in range(first + 1, size):
                    subsets.append((members[first], members[second]))
                    for third in range(second + 1, size):
                        subsets.append((members[first], members[second], members[third]))
        return subsets


def objective_score(plan, objective):
    if objective == TrimObjective.LONGEST:
        return plan.longest_segment_mm, plan.retained_length_mm
    return plan.retained_length_mm, plan.longest_segment_mm


def interval_signature(plan):
    return tuple((region.start_row, region.end_row) for region in plan.cut_regions)


def restamp(plan, objective):
    return TrimPlan(
        objective.value,
        plan.feasible,
        plan.cut_count,
        plan.removed_length_mm,
        plan.retained_length_mm,
        plan.longest_segment_mm,
        plan.shippable_length_mm,
        plan.shippable_segment_count,
        plan.selected_spots,
        plan.cut_regions,
        plan.segments,
        plan.retain,
        plan.conflicts,
        plan.warnings,
        plan.total_length_mm,
    )


def evaluate_trim(context, payload):
    request = TrimRequest.parse(payload)
    planner = TrimPlanner(context)
    subsets = planner.feasible_subsets(request)

    unique = {}
    baseline = None
    for ids in subsets:
        sub_request = TrimRequest(
            request.margin_mm,
            request.minimum_length_mm,
            request.retain_start_mm,
            request.retain_end_mm,
            ids,
        )
        plan = planner.plan(TrimObjective.LONGEST, sub_request)
        if not plan.feasible:
            continue
        signature = interval_signature(plan)
        prior = unique.get(signature)
        if prior is None or len(plan.selected_spots) > len(prior.selected_spots):
            unique[signature] = plan
        if not ids:
            baseline = plan

    removal_plans = [plan for plan in unique.values() if plan.selected_spots]
    if not removal_plans:
        fallback = baseline or planner.plan(TrimObjective.LONGEST, request)
        requested = None
        if request.spot_ids:
            requested = planner.plan(TrimObjective.LONGEST, request).as_dict()
        return {
            "longest_segment": fallback.as_dict(),
            "max_retained": restamp(fallback, TrimObjective.RETAIN).as_dict(),
            "objectives_differ": False,
            "candidate_count": 0,
            "available_spots": len(planner.spots),
            "cut_limit": CUT_LIMIT,
            "max_removed_spots": MAX_REMOVED_SPOTS,
            "baseline": baseline.as_dict() if baseline is not None else None,
            "requested": requested,
        }

    longest = max(removal_plans, key=lambda plan: objective_score(plan, TrimObjective.LONGEST))
    retained = max(removal_plans, key=lambda plan: objective_score(plan, TrimObjective.RETAIN))
    differ = interval_signature(longest) != interval_signature(retained)
    response = {
        "longest_segment": restamp(longest, TrimObjective.LONGEST).as_dict(),
        "max_retained": restamp(retained, TrimObjective.RETAIN).as_dict(),
        "objectives_differ": differ,
        "candidate_count": len(removal_plans),
        "available_spots": len(planner.spots),
        "cut_limit": CUT_LIMIT,
        "max_removed_spots": MAX_REMOVED_SPOTS,
        "baseline": baseline.as_dict() if baseline is not None else None,
    }
    if request.spot_ids:
        requested = planner.plan(TrimObjective.LONGEST, request)
        response["requested"] = requested.as_dict()
    else:
        response["requested"] = None
    return response
