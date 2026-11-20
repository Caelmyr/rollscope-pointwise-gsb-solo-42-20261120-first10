import io
import os
from pathlib import Path

from flask import Blueprint, Flask, Response, current_app, jsonify, render_template, request, send_file
from werkzeug.exceptions import BadRequest, HTTPException, RequestEntityTooLarge

from . import __version__
from .analysis import SECTION_TITLES
from .archive import MAX_PROJECT_BYTES
from .capture_codec import MAX_BUNDLE_BYTES
from .demo import available_demos
from .domain import (
    MissingRecord,
    RevisionConflict,
    UserInputError,
    clean_text,
    exact_keys,
    finite_number,
    read_enum,
    whole_number,
)
from .raster import Layer
from .repository import Repository
from .service import CaptureService


PROJECT_DIR = Path(__file__).resolve().parent.parent
api = Blueprint("api", __name__, url_prefix="/api")
PAGE_LIMIT = 950


def capture_service():
    return current_app.extensions["rollscope"]


def body(fields, with_revision=False):
    if not request.is_json:
        raise UserInputError("Send a JSON object.")
    incoming = request.get_json()
    if not isinstance(incoming, dict):
        raise UserInputError("Send a JSON object.")
    required = (*fields, "revision") if with_revision else fields
    exact_keys(incoming, required, "Request")
    if with_revision:
        revision = whole_number(incoming.pop("revision"), "Revision", 0, 2**31 - 1)
        return incoming, revision
    return incoming


def integer_argument(name, default, minimum, maximum):
    value = request.args.get(name)
    if value is None:
        return default
    try:
        converted = int(value)
    except ValueError as error:
        raise UserInputError(f"{name} must be a whole number.") from error
    return whole_number(converted, name, minimum, maximum)


def decimal_argument(name, minimum, maximum):
    value = request.args.get(name)
    if value is None:
        raise UserInputError(f"Provide a {name} coordinate.")
    try:
        converted = float(value)
    except ValueError as error:
        raise UserInputError(f"{name} must be numeric.") from error
    return finite_number(converted, name, minimum, maximum)


def revision_argument():
    return integer_argument("revision", None, 0, 2**31 - 1)


def points_argument():
    return integer_argument("points", 280, 24, PAGE_LIMIT)


def layer_argument():
    return read_enum(Layer, request.args.get("layer", "station"), "Image layer")


def window_argument():
    return request.args.get("window")


def validate_section(section):
    if not isinstance(section, str) or section not in SECTION_TITLES:
        raise MissingRecord("That analysis section does not exist.")
    return section


def stream_attachment(data, name, media_type):
    return send_file(
        io.BytesIO(data),
        mimetype=media_type,
        as_attachment=True,
        download_name=name,
        max_age=0,
    )


@api.get("/health")
def health():
    capture_service().repository.health()
    return {"status": "ok", "product": "RollScope", "version": __version__}


@api.get("/captures")
def captures():
    return {"captures": capture_service().repository.list_captures(), "demos": available_demos()}


@api.post("/captures/demo")
def make_demo():
    data = body(("profile",))
    capture = capture_service().make_demo(data["profile"])
    return capture.manifest(), 201


@api.post("/captures/import")
def import_capture():
    uploaded = request.files.get("capture")
    if uploaded is None or not uploaded.filename:
        raise UserInputError("Select a capture NPZ file.")
    data = uploaded.stream.read(MAX_BUNDLE_BYTES + 1)
    record = capture_service().import_bundle(data, uploaded.filename)
    return record.manifest(), 201


@api.post("/captures/restore")
def restore_project():
    uploaded = request.files.get("project")
    if uploaded is None or not uploaded.filename:
        raise UserInputError("Select a RollScope project archive.")
    payload = uploaded.stream.read(MAX_PROJECT_BYTES + 1)
    record = capture_service().import_project(payload)
    return record.manifest(), 201


@api.get("/captures/<ref>/project")
def export_project(ref):
    payload = capture_service().export_project(ref)
    return stream_attachment(payload, f"rollscope-{ref}-project.zip", "application/zip")


@api.get("/captures/<ref>")
def capture(ref):
    return capture_service().capture_details(ref)


@api.get("/captures/<ref>/source")
def source(ref):
    service = capture_service()
    payload = service.repository.source_bundle(ref)
    return stream_attachment(payload, f"rollscope-{ref}-capture.npz", "application/octet-stream")


@api.get("/captures/<ref>/analysis/<section>")
def analysis(ref, section):
    return capture_service().inspect(
        ref, revision_argument(), validate_section(section), points_argument()
    )


@api.get("/captures/<ref>/analysis/<section>/export")
def analysis_export(ref, section):
    payload = capture_service().export_analysis(ref, revision_argument(), validate_section(section))
    return stream_attachment(payload, f"rollscope-{ref}-{section}.zip", "application/zip")


@api.get("/captures/<ref>/preview.png")
def raster(ref):
    overlay = request.args.get("overlay", "1") != "0"
    content, mapping, revision = capture_service().raster(
        ref, revision_argument(), layer_argument(), overlay, window_argument()
    )
    response = Response(content, mimetype="image/png")
    response.headers["X-Review-Revision"] = str(revision)
    response.headers["X-Preview-Shape"] = f"{mapping['display_width']}x{mapping['display_height']}"
    response.cache_control.no_store = True
    return response


@api.get("/captures/<ref>/view")
def view(ref):
    return capture_service().view(ref, revision_argument(), layer_argument(), window_argument())


@api.get("/captures/<ref>/probe")
def probe(ref):
    return capture_service().probe(
        ref,
        revision_argument(),
        layer_argument(),
        window_argument(),
        decimal_argument("x", 0, 1),
        decimal_argument("y", 0, 1),
    )


@api.get("/captures/<ref>/profile")
def profile(ref):
    return capture_service().profile(
        ref,
        revision_argument(),
        layer_argument(),
        window_argument(),
        decimal_argument("y", 0, 1),
        points_argument(),
    )


@api.post("/captures/<ref>/ruler")
def ruler(ref):
    endpoints, revision = body(("start", "end"), with_revision=True)
    return capture_service().measure(ref, revision, endpoints, points_argument())


@api.post("/captures/<ref>/ruler/export")
def ruler_export(ref):
    endpoints, revision = body(("start", "end"), with_revision=True)
    payload = capture_service().export_measurement(ref, revision, endpoints)
    return stream_attachment(payload, f"rollscope-{ref}-ruler.zip", "application/zip")


@api.post("/captures/<ref>/settings")
def settings(ref):
    data, revision = body(("section", "values"), with_revision=True)
    return capture_service().patch_settings(ref, revision, data)


@api.put("/captures/<ref>/notes")
def notes(ref):
    data, revision = body(("notes",), with_revision=True)
    return {"state": capture_service().set_notes(ref, revision, data["notes"])}


@api.get("/captures/<ref>/alignment")
def alignment(ref):
    return capture_service().alignment(ref, revision_argument())


@api.post("/captures/<ref>/alignment/parse")
def alignment_parse(ref):
    data = body(("log_text",))
    return capture_service().preview_alignment(ref, data)


@api.put("/captures/<ref>/alignment")
def alignment_save(ref):
    data, revision = body(("log_text", "anchors", "certainty_mm"), with_revision=True)
    return capture_service().save_alignment(ref, revision, data)


@api.delete("/captures/<ref>/alignment")
def alignment_clear(ref):
    _, revision = body((), with_revision=True)
    return capture_service().clear_alignment(ref, revision)


@api.post("/captures/<ref>/windows")
def create_window(ref):
    data, revision = body(("label", "note", "status", "region"), with_revision=True)
    return capture_service().write_window(ref, revision, data), 201


@api.put("/captures/<ref>/windows/<window_id>")
def update_window(ref, window_id):
    data, revision = body(("label", "note", "status", "region"), with_revision=True)
    return capture_service().write_window(ref, revision, data, window_id)


@api.delete("/captures/<ref>/windows/<window_id>")
def delete_window(ref, window_id):
    _, revision = body((), with_revision=True)
    return {"state": capture_service().delete_window(ref, revision, window_id)}


@api.get("/captures/<ref>/windows/<window_id>")
def window_result(ref, window_id):
    return capture_service().window_result(ref, revision_argument(), window_id)


@api.get("/captures/<ref>/windows/<window_id>/export")
def window_export(ref, window_id):
    result = capture_service().window_export(ref, revision_argument(), window_id)
    return stream_attachment(result, f"rollscope-{ref}-window-{window_id}.zip", "application/zip")


@api.post("/captures/<ref>/snapshots")
def create_snapshot(ref):
    payload, revision = body(("section", "label"), with_revision=True)
    label = clean_text(payload["label"], "Snapshot name", 75)
    section = validate_section(payload["section"])
    return capture_service().save_snapshot(ref, revision, section, label), 201


@api.get("/captures/<ref>/snapshots/<snapshot_id>")
def snapshot(ref, snapshot_id):
    return capture_service().snapshot(ref, snapshot_id)


@api.get("/captures/<ref>/snapshots/<snapshot_id>/compare")
def compare_snapshot(ref, snapshot_id):
    return capture_service().compare_snapshot(ref, revision_argument(), snapshot_id)


@api.get("/captures/<ref>/snapshots/<snapshot_id>/export")
def export_snapshot(ref, snapshot_id):
    payload = capture_service().export_snapshot(ref, snapshot_id)
    return stream_attachment(
        payload, f"rollscope-{ref}-snapshot-{snapshot_id}.zip", "application/zip"
    )


def api_error(error, status, **extra):
    return jsonify({"error": str(error), **extra}), status


def create_app(data_dir=None, seed=True):
    app = Flask(
        __name__,
        template_folder=str(PROJECT_DIR / "templates"),
        static_folder=str(PROJECT_DIR / "static"),
    )
    app.config.update(MAX_CONTENT_LENGTH=MAX_BUNDLE_BYTES + 1024 * 1024)
    storage = data_dir or os.environ.get("ROLLSCOPE_DATA_DIR", PROJECT_DIR / "data")
    service = CaptureService(Repository(storage))
    if seed:
        service.seed_demo()
    app.extensions["rollscope"] = service
    app.register_blueprint(api)

    @app.get("/")
    def workspace():
        return render_template("index.html", version=__version__)

    @app.get("/guide")
    def guide():
        return render_template("guide.html", version=__version__)

    @app.after_request
    def defensive_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob: data:; object-src 'none'; base-uri 'none'; "
            "form-action 'self'; frame-ancestors 'none'"
        )
        return response

    @app.errorhandler(UserInputError)
    def rejected(error):
        return api_error(error, 400)

    @app.errorhandler(BadRequest)
    def bad_json(error):
        return api_error("The request could not be read. Verify its format.", 400)

    @app.errorhandler(RevisionConflict)
    def conflict(error):
        return api_error(error, 409, current_revision=error.revision)

    @app.errorhandler(MissingRecord)
    def missing(error):
        return api_error(error, 404)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return api_error("The uploaded file exceeds the supported size.", 413)

    @app.errorhandler(HTTPException)
    def error_page(error):
        if request.path.startswith("/api/"):
            return api_error(error.description, error.code)
        return error

    return app
