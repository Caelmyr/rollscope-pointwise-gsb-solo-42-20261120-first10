"""Project a time-stamped machine operation log onto capture stations.

The acquisition camera and the machine control cabinet run separate clocks,
and the cabinet clock may be rolled back during a shift. Record order in the
log stays truthful even when cabinet timestamps repeat, so the log is split
into clock legs at each rollback. Inside one leg the cabinet clock advances
with real time, so a linear map links cabinet time to material station. User
anchors (known operations tied to narrow station ranges) confine that map to
a convex family; every other record is projected to the envelope of stations
the whole family allows. Narrow envelopes are confident placements, while
wide envelopes mean several alignments explain the record equally well.
"""

from datetime import datetime, timezone
from itertools import combinations

import numpy as np

from .domain import UserInputError, clean_text
from .records import MAX_LOG_EVENTS, MAX_LOG_TEXT, LogEvent


ROLLBACK_EPS_S = 1e-6
SPEED_FLOOR_MM_S = 1e-3
SPEED_CEILING_MM_S = 1e4

DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)
CLOCK_FORMATS = ("%H:%M:%S.%f", "%H:%M:%S", "%H:%M")

LEG_PROBLEMS = {
    "no_anchors": "has no anchors",
    "conflicting": "has anchors that contradict each other",
    "underdetermined": "does not have enough anchor information",
}


def parse_instant(text, line):
    value = text.strip()
    if not value:
        raise UserInputError(f"Log line {line}: the time value is empty.")
    try:
        moment = float(value)
    except ValueError:
        moment = None
    if moment is not None:
        if not -1e12 <= moment <= 1e12:
            raise UserInputError(
                f"Log line {line}: the seconds value is outside the supported range."
            )
        return moment, "seconds"
    for date_format in DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(value, date_format)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone.utc).timestamp(), "datetime"
    for clock_format in CLOCK_FORMATS:
        try:
            parsed = datetime.strptime(value, clock_format)
        except ValueError:
            continue
        seconds = parsed.hour * 3600 + parsed.minute * 60 + parsed.second
        return seconds + parsed.microsecond / 1e6, "clock"
    raise UserInputError(
        f"Log line {line}: cannot read '{value}'. "
        "Use HH:MM:SS, YYYY-MM-DD HH:MM:SS, or plain seconds."
    )


def parse_log_text(text):
    log_text = clean_text(text, "Operation log", MAX_LOG_TEXT)
    events = []
    time_forms = set()
    for number, raw in enumerate(log_text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [piece.strip() for piece in line.split(",", 3)]
        if len(parts) < 3:
            raise UserInputError(f"Log line {number}: expected 'kind, start, end[, note]'.")
        kind = clean_text(parts[0], f"Log line {number} event kind", 40)
        note = ""
        if len(parts) > 3:
            note = clean_text(parts[3], f"Log line {number} note", 200, True)
        start_s, form = parse_instant(parts[1], number)
        end_s, _ = parse_instant(parts[2], number)
        if end_s < start_s:
            raise UserInputError(
                f"Log line {number}: the end is earlier than the start. If this record "
                "straddles the clock rollback, split it at the rollback and import both halves."
            )
        time_forms.add(form)
        events.append(
            LogEvent(
                f"e{len(events) + 1}", kind, note, parts[1], parts[2], start_s, end_s
            )
        )
    if not events:
        raise UserInputError("The operation log contains no records.")
    if len(events) > MAX_LOG_EVENTS:
        raise UserInputError(
            f"The operation log may hold at most {MAX_LOG_EVENTS} records."
        )
    warnings = []
    if "datetime" in time_forms and len(time_forms) > 1:
        warnings.append(
            "The log mixes full datetimes with shorter time forms; "
            "verify every record shares one time reference."
        )
    return tuple(events), warnings


def leg_assignment(events):
    """Clock leg per record: a start earlier than its predecessor opens a new leg."""
    legs = []
    current = 0
    previous = None
    for event in events:
        if previous is not None and event.start_s < previous - ROLLBACK_EPS_S:
            current += 1
        legs.append(current)
        previous = event.start_s
    return legs


def leg_title(index, count):
    if count == 1:
        return "Leg 1"
    if count == 2:
        return "Before rollback" if index == 0 else "After rollback"
    return f"Leg {index + 1} of {count}"


def anchor_constraints(pairs, reference):
    """Half-planes a·v + b·c <= d for s(t) = v·(t - reference) + c."""
    constraints = []
    for event, anchor in pairs:
        start = event.start_s - reference
        end = event.end_s - reference
        low, high = anchor.station_start_mm, anchor.station_end_mm
        constraints.append((-start, -1.0, -low))
        constraints.append((start, 1.0, high))
        constraints.append((-end, -1.0, -low))
        constraints.append((end, 1.0, high))
    return constraints


def feasible_vertices(constraints):
    """Vertices of the convex (v, c) family honouring every constraint."""
    lines = [
        *constraints,
        (-1.0, 0.0, -SPEED_FLOOR_MM_S),
        (1.0, 0.0, SPEED_CEILING_MM_S),
    ]
    scale = max(1.0, max(abs(constant) for _, _, constant in lines))
    tolerance = 1e-9 * scale + 1e-6
    vertices = []
    for (a1, b1, d1), (a2, b2, d2) in combinations(lines, 2):
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-9:
            continue
        v = (d1 * b2 - d2 * b1) / det
        c = (a1 * d2 - a2 * d1) / det
        if all(a * v + b * c <= d + tolerance for a, b, d in lines):
            vertices.append((round(v, 6), round(c, 3)))
    return sorted(set(vertices))


def conflicting_pair(pairs, reference):
    for first, second in combinations(pairs, 2):
        if not feasible_vertices(anchor_constraints((first, second), reference)):
            return first[0].id, second[0].id
    return None


def solve_leg(pairs):
    if not pairs:
        return {"status": "no_anchors", "reference": 0.0, "vertices": [], "conflict": None}
    reference = min(event.start_s for event, _ in pairs)
    vertices = feasible_vertices(anchor_constraints(pairs, reference))
    if not vertices:
        return {
            "status": "conflicting",
            "reference": reference,
            "vertices": [],
            "conflict": conflicting_pair(pairs, reference),
        }
    speeds = [v for v, _ in vertices]
    if (
        min(speeds) <= SPEED_FLOOR_MM_S + 1e-9
        or max(speeds) >= SPEED_CEILING_MM_S - 1e-6
    ):
        return {
            "status": "underdetermined",
            "reference": reference,
            "vertices": vertices,
            "conflict": None,
        }
    return {"status": "aligned", "reference": reference, "vertices": vertices, "conflict": None}


def envelope(solution, start_s, end_s):
    reference = solution["reference"]
    lo = min(v * (start_s - reference) + c for v, c in solution["vertices"])
    hi = max(v * (end_s - reference) + c for v, c in solution["vertices"])
    return lo, hi


def measured_speed(capture, low, high):
    """Median transport speed measured by the camera near the given stations."""
    stations = capture.stations
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.diff(stations) / np.diff(capture.seconds)
    valid = np.isfinite(speed) & (speed > 0)
    pad = 0.1 * max(high - low, 1.0)
    near = valid & (stations[:-1] >= low - pad) & (stations[:-1] <= high + pad)
    chosen = speed[near] if np.any(near) else speed[valid]
    if not len(chosen):
        return None
    return float(np.median(chosen))


def speed_deviation(speeds, measured):
    if measured is None or measured <= 0:
        return 0.0
    low, high = min(speeds), max(speeds)
    if low <= measured <= high:
        return 0.0
    return min(abs(measured - low), abs(measured - high)) / measured


def check_boundary(number, legs, solutions, measured, placements, pitch_mm):
    before = legs[0]
    after = legs[1]
    title_before = leg_title(before, legs[1] + 1)
    title_after = leg_title(after, legs[1] + 1)
    base = {
        "index": number - 1,
        "title": f"Rollback {number}",
        "between": [title_before, title_after],
    }
    problems = [
        f"{title} {LEG_PROBLEMS[solutions[leg]['status']]}"
        for leg, title in ((before, title_before), (after, title_after))
        if solutions[leg]["status"] != "aligned"
    ]
    if problems:
        return base | {
            "status": "unverifiable",
            "station_drop_mm": None,
            "suspect_leg": None,
            "suspect_title": None,
            "detail": f"Rollback {number} cannot be checked: " + "; ".join(problems) + ".",
        }
    first, second = placements
    drop = first["station_start_mm"] - second["station_end_mm"]
    if drop <= 2 * pitch_mm:
        gap = second["station_start_mm"] - first["station_end_mm"]
        return base | {
            "status": "consistent",
            "station_drop_mm": drop,
            "suspect_leg": None,
            "suspect_title": None,
            "detail": (
                f"Rollback {number} lines up: material order is preserved across the "
                f"rollback (station gap {gap:.1f} mm between the neighbouring records)."
            ),
        }
    deviations = {
        leg: speed_deviation(
            [v for v, _ in solutions[leg]["vertices"]], measured[leg]
        )
        for leg in (before, after)
    }
    if abs(deviations[before] - deviations[after]) < 1e-9:
        suspect = None
        advice = (
            "Both sides imply speeds compatible with the measured transport, so one "
            "anchor station range is off; recheck the anchors on both sides."
        )
    else:
        suspect = max(deviations, key=deviations.get)
        speeds = [v for v, _ in solutions[suspect]["vertices"]]
        advice = (
            f"The {leg_title(suspect, legs[1] + 1).lower()} anchors imply "
            f"{min(speeds):.1f}–{max(speeds):.1f} mm/s against "
            f"{measured[suspect]:.1f} mm/s measured on the capture nearby, a larger "
            f"mismatch than the other side. Recheck the "
            f"{leg_title(suspect, legs[1] + 1).lower()} anchors."
        )
    return base | {
        "status": "inconsistent",
        "station_drop_mm": drop,
        "suspect_leg": suspect,
        "suspect_title": None if suspect is None else leg_title(suspect, legs[1] + 1),
        "detail": (
            f"Rollback {number} does not line up: stations move backward by "
            f"{drop:.1f} mm across the rollback. " + advice
        ),
    }


def build_report(capture, grid, gaps, alignment):
    events = alignment.events
    legs = leg_assignment(events)
    leg_count = legs[-1] + 1
    anchors = {anchor.event_id: anchor for anchor in alignment.anchors}
    pairs_by_leg = [[] for _ in range(leg_count)]
    for index, event in enumerate(events):
        anchor = anchors.get(event.id)
        if anchor is not None:
            pairs_by_leg[legs[index]].append((event, anchor))
    solutions = [solve_leg(pairs) for pairs in pairs_by_leg]
    measured = []
    for pairs in pairs_by_leg:
        if pairs:
            low = min(anchor.station_start_mm for _, anchor in pairs)
            high = max(anchor.station_end_mm for _, anchor in pairs)
            measured.append(measured_speed(capture, low, high))
        else:
            measured.append(None)

    cap0 = float(grid.stations[0])
    cap1 = float(grid.stations[-1])
    placements = []
    for index, event in enumerate(events):
        leg = legs[index]
        solution = solutions[leg]
        base = {
            "event_id": event.id,
            "kind": event.kind,
            "start_text": event.start_text,
            "end_text": event.end_text,
            "leg": leg,
        }
        if solution["status"] != "aligned":
            placements.append(
                base
                | {
                    "status": "unplaced",
                    "reason": solution["status"],
                    "station_mm": None,
                    "station_start_mm": None,
                    "station_end_mm": None,
                    "width_mm": None,
                    "crosses_gap": False,
                }
            )
            continue
        low, high = envelope(solution, event.start_s, event.end_s)
        if high < cap0 or low > cap1:
            status = "outside"
        else:
            status = "confident" if high - low <= alignment.certainty_mm else "ambiguous"
        placements.append(
            base
            | {
                "status": status,
                "reason": None,
                "station_start_mm": low,
                "station_end_mm": high,
                "station_mm": (low + high) / 2,
                "width_mm": high - low,
                "crosses_gap": any(
                    not (gap.to_mm < low or gap.from_mm > high) for gap in gaps
                ),
            }
        )

    leg_rows = []
    for leg, solution in enumerate(solutions):
        row = {
            "index": leg,
            "title": leg_title(leg, leg_count),
            "event_count": sum(1 for item in legs if item == leg),
            "anchor_count": len(pairs_by_leg[leg]),
            "status": solution["status"],
            "measured_speed_mm_s": measured[leg],
            "speed_range_mm_s": None,
            "conflict": None,
        }
        if solution["status"] == "aligned":
            speeds = [v for v, _ in solution["vertices"]]
            row["speed_range_mm_s"] = [min(speeds), max(speeds)]
        if solution["status"] == "conflicting":
            if solution["conflict"]:
                first, second = solution["conflict"]
                detail = f"The anchors on {first} and {second} contradict each other."
                row["conflict"] = {"events": [first, second], "detail": detail}
            else:
                detail = "The anchors in this leg contradict each other."
                row["conflict"] = {"events": [], "detail": detail}
        leg_rows.append(row)

    boundaries = []
    number = 0
    for index in range(1, len(events)):
        if legs[index] == legs[index - 1]:
            continue
        number += 1
        boundaries.append(
            check_boundary(
                number,
                (legs[index - 1], legs[index]),
                solutions,
                measured,
                (placements[index - 1], placements[index]),
                grid.row_pitch_mm,
            )
        )

    counts = {}
    for placement in placements:
        counts[placement["status"]] = counts.get(placement["status"], 0) + 1
    return {
        "event_count": len(events),
        "anchor_count": len(alignment.anchors),
        "leg_count": leg_count,
        "rollback_count": leg_count - 1,
        "certainty_mm": alignment.certainty_mm,
        "capture_station_mm": [cap0, cap1],
        "status_counts": counts,
        "legs": leg_rows,
        "boundaries": boundaries,
        "placements": placements,
        "gaps": [{"from_mm": gap.from_mm, "to_mm": gap.to_mm} for gap in gaps],
    }
