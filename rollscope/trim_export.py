from .evidence import Evidence
from .raster import Layer, RasterSpec, png_bytes, preview_image
from .records import TrimPlanRecord


def trim_view(context, plan):
    spec = RasterSpec(Layer.TRIM, trim_plan=plan)
    image, mapping = preview_image(context, spec)
    return image, mapping


def export_trim_plan(record, context, saved: TrimPlanRecord):
    plan = saved.plan
    evidence = Evidence("trim")
    document = {
        "trim_plan": saved.as_dict(),
        "capture": {
            "id": record.id,
            "name": record.capture.metadata.name,
            "material": record.capture.metadata.material,
            "sha256": record.sha256,
        },
    }
    evidence.json("trim-plan.json", document)
    evidence.json("confirmed-settings.json", context.settings.as_dict())
    evidence.table("selected-spots.csv", plan["selected_spots"])
    evidence.table("cut-regions.csv", [region_flat(region) for region in plan["cut_regions"]])
    evidence.table("remnant-segments.csv", plan["segments"])
    cuts = []
    for region in plan["cut_regions"]:
        cuts.append(region["start_cut"] | {"role": "start", "region_end_mm": region["end_mm"]})
        cuts.append(region["end_cut"] | {"role": "end", "region_start_mm": region["start_mm"]})
    evidence.table("cut-positions.csv", cuts)
    if plan["retain"] is not None:
        evidence.table("retained-sample.csv", [plan["retain"]])
    image, mapping = trim_view(context, plan)
    evidence.add("trim-overview.png", png_bytes(image))
    evidence.json("raster-mapping.json", mapping.as_dict())
    return evidence.make(
        {
            "capture_sha256": record.sha256,
            "project_capture_id": record.id,
            "revision": record.state.revision,
            "trim_plan_id": saved.id,
            "objective": saved.objective,
        }
    )


def region_flat(region):
    return {
        "start_mm": region["start_mm"],
        "end_mm": region["end_mm"],
        "start_row": region["start_row"],
        "end_row": region["end_row"],
        "length_mm": region["length_mm"],
        "spot_ids": region["spot_ids"],
        "start_cut_station_mm": region["start_cut"]["station_mm"],
        "start_cut_coverage": region["start_cut"]["coverage"],
        "end_cut_station_mm": region["end_cut"]["station_mm"],
        "end_cut_coverage": region["end_cut"]["coverage"],
    }
