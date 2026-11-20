import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .domain import StationWindow
from .raster import Layer, RasterSpec, preview_image
from .window_review import WindowCells, review_window


PAGE_WIDTH = 1560
PADDING = 66
INK = "#18212b"
MUTED = "#596571"
RULE = "#dae0e4"
ACCENT = "#166556"
AMBER = "#8b420c"


@dataclass(frozen=True)
class Panel:
    region: StationWindow
    station_from: float
    station_to: float
    pixel_width: int
    pixel_height: int


def font(size):
    return ImageFont.load_default(size=size)


def text_width(draw, value, face):
    return draw.textlength(str(value), font=face)


def fit_label(draw, value, face, maximum):
    value = str(value)
    if text_width(draw, value, face) <= maximum:
        return value
    tail = "..."
    while value and text_width(draw, value + tail, face) > maximum:
        value = value[:-1]
    return value + tail


def wrapped_lines(draw, value, face, width):
    for paragraph in value.splitlines():
        current = ""
        for word in paragraph.split():
            next_line = f"{current} {word}" if current else word
            if text_width(draw, next_line, face) <= width:
                current = next_line
                continue
            if current:
                yield current
            current = fit_label(draw, word, face, width)
        if current:
            yield current


def format_scalar(value, suffix="", decimals=2):
    if value is None:
        return "Unavailable"
    return f"{value:,.{decimals}f}{suffix}"


def panel_plan(grid, cells, maximum=4):
    row_count = cells.rows.stop - cells.rows.start
    column_count = cells.columns.stop - cells.columns.start
    cross_size = float(cells.transverse[-1] - cells.transverse[0])
    along_size = grid.row_pitch_mm * row_count
    count = math.ceil(along_size / (3.6 * cross_size))
    count = max(1, min(maximum, count))
    width_budget = (PAGE_WIDTH - PADDING * 2 - 40 * (count - 1)) // count
    width_budget = min(780, width_budget - 92)
    scale = min(
        width_budget / cross_size,
        (650 * count) / along_size,
        16,
    )
    scale = max(scale, min(1.0, width_budget / cross_size))
    edges = np.linspace(cells.rows.start, cells.rows.stop, count + 1, dtype=np.int64)
    panels = []
    for first, stop in zip(edges[:-1], edges[1:]):
        if stop <= first:
            continue
        y0 = float(grid.stations[first])
        y1 = float(grid.stations[stop - 1])
        x0 = float(cells.transverse[0])
        x1 = float(cells.transverse[-1])
        if y1 == y0:
            approximation = max(1e-8, grid.row_pitch_mm / 4)
            lower = y0 - approximation
            upper = y1 + approximation
        else:
            lower, upper = y0, y1
        if x0 == x1:
            approximation = max(1e-8, grid.column_pitch_mm / 4)
            x0 -= approximation
            x1 += approximation
        panels.append(
            Panel(
                StationWindow(lower, upper, x0, x1),
                y0,
                y1,
                max(1, int(round(cross_size * scale))),
                max(1, min(700, int(round((stop - first) * grid.row_pitch_mm * scale)))),
            )
        )
    return panels


def nice_step(span, positions):
    if span <= 0:
        return 0.0
    rough = span / positions
    magnitude = math.pow(10, math.floor(math.log10(rough)))
    for candidate in (1, 2, 2.5, 5, 10):
        if candidate * magnitude >= rough:
            return candidate * magnitude
    return magnitude * 10


def station_ticks(lower, upper):
    if lower == upper:
        return [(lower, 0.5)]
    span = upper - lower
    step = nice_step(span, 5)
    start = math.ceil(lower / step) * step
    values = np.arange(start, upper + step * 1e-8, step)
    if not values.size:
        return [(lower, 0.0), (upper, 1.0)]
    return [(float(value), float((value - lower) / span)) for value in values]


def measurement_cards(report):
    summary = report["summary"]
    return (
        ("USABLE GRID CELLS", format_scalar(summary["known_fraction"] * 100, "%", 1)),
        ("ENCODER GAPS", str(summary["gap_count"])),
        ("OBSERVED MATERIAL", format_scalar(summary["observed_material_area_mm2"], " mm²")),
        ("SPOT FOOTPRINTS", str(len(report["spots"]))),
    )


def paint_header(draw, record, window, report):
    draw.text((PADDING, 39), "RollScope / WINDOW INSPECTION", fill=ACCENT, font=font(20))
    header = fit_label(draw, window.label, font(46), PAGE_WIDTH - 2 * PADDING)
    draw.text((PADDING, 77), header, fill=INK, font=font(46))
    source = f"{record.capture.metadata.name}  /  {record.capture.metadata.material}"
    if record.capture.metadata.synthetic:
        source += "  /  Synthetic capture"
    draw.text((PADDING, 139), fit_label(draw, source, font(23), PAGE_WIDTH - 2 * PADDING),
              fill=MUTED, font=font(23))
    series = measurement_cards(report)
    section_width = (PAGE_WIDTH - 2 * PADDING) // len(series)
    for index, (label, value) in enumerate(series):
        x = PADDING + index * section_width
        if index:
            draw.line((x - 20, 212, x - 20, 284), fill=RULE, width=2)
        draw.text((x, 212), label, fill=MUTED, font=font(17))
        draw.text((x, 241), value, fill=INK, font=font(33))
    draw.line((PADDING, 310, PAGE_WIDTH - PADDING, 310), fill=RULE, width=2)


def paint_scale(draw, panel, x, top):
    axis = x + panel.pixel_width
    for station, ratio in station_ticks(panel.station_from, panel.station_to):
        point = top + round(ratio * max(0, panel.pixel_height - 1))
        draw.line((axis, point, axis + 8, point), fill=MUTED, width=1)
        draw.text((axis + 12, point), f"{station:.2f}", fill=MUTED, font=font(15), anchor="lm")


def paint_panel(sheet, draw, panel, context, x, top, index):
    spec = RasterSpec(Layer.SPOTS, window=panel.region)
    preview, _ = preview_image(context, spec)
    shape = panel.pixel_width, panel.pixel_height
    image = preview.resize(shape, resample=Image.Resampling.NEAREST)
    sheet.paste(image, (x, top))
    draw.rectangle(
        (x, top, x + panel.pixel_width, top + panel.pixel_height),
        outline=RULE,
        width=1,
    )
    draw.text((x, top - 34), f"Strip {index} / station mm", fill=INK, font=font(18))
    paint_scale(draw, panel, x, top)
    cross_text = (
        f"{panel.region.transverse_start_mm:.2f} to "
        f"{panel.region.transverse_end_mm:.2f} mm across"
    )
    draw.text((x, top + panel.pixel_height + 12), cross_text, fill=MUTED, font=font(16))


def paint_details(draw, record, window, report, top):
    draw.line((PADDING, top, PAGE_WIDTH - PADDING, top), fill=RULE, width=2)
    x = PADDING
    y = top + 24
    draw.text((x, y), "REVIEW NOTE", fill=ACCENT, font=font(18))
    note = window.note.strip() or "No note entered for this window."
    for line in list(wrapped_lines(draw, note, font(21), 860))[:5]:
        y += 30
        draw.text((x, y), line, fill=INK, font=font(21))
    x = 1030
    count = report["summary"]["contributing_source_lines"]
    details = [
        f"Window status: {window.status.value}",
        f"Source scan lines: {count}",
        f"Revision: {record.state.revision}",
        f"Capture SHA-256: {record.sha256[:16]}…",
    ]
    for index, detail in enumerate(details):
        draw.text((x, top + 24 + index * 28), detail, fill=MUTED, font=font(18))


def window_sheet(record, context, window):
    report = review_window(context, window)
    cells = WindowCells.take(context, window.region)
    panels = panel_plan(context.grid, cells)
    maximum_height = max(panel.pixel_height for panel in panels)
    start_y = 391
    details_top = start_y + maximum_height + 89
    sheet = Image.new("RGB", (PAGE_WIDTH, details_top + 235), "#f9fbfc")
    draw = ImageDraw.Draw(sheet)
    paint_header(draw, record, window, report)
    lane_width = (PAGE_WIDTH - 2 * PADDING) // len(panels)
    for index, panel in enumerate(panels, 1):
        x = PADDING + lane_width * (index - 1)
        paint_panel(sheet, draw, panel, context, x, start_y, index)
    paint_details(draw, record, window, report, details_top)
    draw.text(
        (PADDING, details_top + 194),
        "Orange: darker contrast    Teal: brighter contrast    Hatched: unavailable support",
        fill=AMBER,
        font=font(18),
    )
    return sheet
