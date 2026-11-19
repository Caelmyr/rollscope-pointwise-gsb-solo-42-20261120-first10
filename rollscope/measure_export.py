from PIL import ImageDraw

from .evidence import Evidence
from .raster import Layer, RasterSpec, png_bytes, preview_image
from .ruler import PathSupport, evaluate_ruler
from .window_review import row_provenance


PATH_COLORS = {
    PathSupport.GAP: (230, 120, 105),
    PathSupport.BAD_PIXEL: (211, 144, 205),
    PathSupport.BACKGROUND: (105, 184, 232),
    PathSupport.MATERIAL: (255, 219, 109),
}


def ruler_view(context, path):
    image, mapping = preview_image(context, RasterSpec(Layer.STATION))
    draw = ImageDraw.Draw(image)
    grid = context.grid
    x_origin = grid.transverse[0]
    y_origin = grid.stations[0]

    def in_image(stations, transverse):
        return mapping.pixel(
            (stations - y_origin) / grid.row_pitch_mm,
            (transverse - x_origin) / grid.column_pitch_mm,
        )

    boundary = 0
    while boundary < len(path.states):
        stop = boundary + 1
        while stop < len(path.states) and path.states[stop] == path.states[boundary]:
            stop += 1
        start_xy = in_image(*path.ruler.interpolate(path.enter[boundary]))
        end_xy = in_image(*path.ruler.interpolate(path.leave[stop - 1]))
        draw.line((start_xy, end_xy), fill=(31, 26, 29), width=5)
        stroke = PATH_COLORS[PathSupport(path.states[boundary])]
        draw.line((start_xy, end_xy), fill=stroke, width=3)
        boundary = stop

    for endpoint in (path.ruler.start, path.ruler.end):
        x, y = in_image(endpoint.station_mm, endpoint.transverse_mm)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 230, 115), outline="#211933")
    return image, mapping


def export_ruler(record, context, ruler):
    path = evaluate_ruler(context, ruler)
    result = path.as_dict()
    evidence = Evidence("ruler")
    evidence.json("measurement.json", {k: v for k, v in result.items() if k != "profile"})
    evidence.json("settings.json", context.settings.as_dict())
    evidence.table("crossed-cells.csv", result["profile"])
    evidence.table("support-intervals.csv", result["intervals"])
    evidence.table("spot-intersections.csv", result["spots"])
    raster, mapping = ruler_view(context, path)
    evidence.add("measurement.png", png_bytes(raster))
    evidence.json("raster-mapping.json", mapping.as_dict())
    first = min(path.rows)
    last = max(path.rows)
    evidence.table(
        "contributing-lines.csv",
        list(row_provenance(context.grid, slice(first, last + 1))),
    )
    return evidence.make(
        {
            "capture_sha256": record.sha256,
            "project_capture_id": record.id,
            "revision": record.state.revision,
        }
    )
