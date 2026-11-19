import hashlib
import io
import json
from zipfile import BadZipFile, ZipFile

import numpy as np
from numpy.lib import format as npformat

from .domain import MAX_COLS, MAX_REFERENCE_ROWS, MAX_ROWS, Capture, CaptureMetadata, UserInputError


MAX_BUNDLE_BYTES = 48 * 1024 * 1024
MAX_HEADER_BYTES = 4096
MAX_METADATA_BYTES = 12000
NPZ_SPEC = {
    "raw": ("iu", ((8, MAX_ROWS), (16, MAX_COLS))),
    "ticks": ("iu", ((8, MAX_ROWS),)),
    "seconds": ("iuf", ((8, MAX_ROWS),)),
    "dark": ("iu", ((1, MAX_REFERENCE_ROWS), (16, MAX_COLS))),
    "white": ("iu", ((1, MAX_REFERENCE_ROWS), (16, MAX_COLS))),
    "metadata_utf8": ("u", ((2, MAX_METADATA_BYTES),)),
}


def canonical_json(document):
    return json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def read_metadata(array):
    if array.dtype != np.uint8:
        raise UserInputError("Capture metadata must be a vector of UTF-8 bytes.")
    try:
        document = json.loads(array.tobytes().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise UserInputError("Capture metadata must be a UTF-8 JSON object.") from error
    return CaptureMetadata.parse(document)


def read_capture(payload):
    if len(payload) > MAX_BUNDLE_BYTES:
        raise UserInputError("The scan bundle exceeds 48 MiB.")
    expected_names = [f"{name}.npy" for name in NPZ_SPEC]
    values = {}
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            if sorted(archive.namelist()) != sorted(expected_names):
                raise UserInputError(
                    "The bundle must contain raw, ticks, seconds, dark, white and metadata_utf8."
                )
            for key, (kinds, dimensional_limits) in NPZ_SPEC.items():
                member = archive.getinfo(f"{key}.npy")
                values[key] = read_npy(archive, member, kinds, dimensional_limits)
    except UserInputError:
        raise
    except (BadZipFile, ValueError, EOFError, RuntimeError, OverflowError, OSError) as error:
        raise UserInputError("The input is not a supported NumPy capture bundle.") from error
    metadata = read_metadata(values.pop("metadata_utf8"))
    return Capture.build(**values, metadata=metadata)


def read_npy(archive, member, supported_kinds, dimensions):
    largest = int(np.prod([limit[1] for limit in dimensions], dtype=np.int64))
    if member.file_size > largest * 8 + MAX_HEADER_BYTES + 16:
        raise UserInputError(f"{member.filename} exceeds its decoded size limit.")
    with archive.open(member) as stream:
        format_version = npformat.read_magic(stream)
        readers = {
            (1, 0): npformat.read_array_header_1_0,
            (2, 0): npformat.read_array_header_2_0,
        }
        if format_version not in readers:
            raise UserInputError(f"{member.filename} uses an unsupported NPY format version.")
        read_header = readers[format_version]
        shape, fortran_order, dtype = read_header(stream, max_header_size=MAX_HEADER_BYTES)
        if len(shape) != len(dimensions):
            raise UserInputError(f"{member.filename} has the wrong number of dimensions.")
        for size, (minimum, maximum) in zip(shape, dimensions):
            if not minimum <= size <= maximum:
                raise UserInputError(f"{member.filename} has an unsupported array shape.")
        if dtype.kind not in supported_kinds or dtype.itemsize not in (1, 2, 4, 8):
            raise UserInputError(f"{member.filename} has an unsupported scalar data type.")
        count = int(np.prod(shape, dtype=np.int64))
        expected_bytes = dtype.itemsize * count
        if stream.tell() + expected_bytes != member.file_size:
            raise UserInputError(f"{member.filename} does not match its declared array shape.")
        raw = stream.read(expected_bytes + 1)
        if len(raw) != expected_bytes:
            raise UserInputError(f"{member.filename} ended before the array was complete.")
        order = "F" if fortran_order else "C"
        return np.frombuffer(raw, dtype=dtype, count=count).reshape(shape, order=order)


def capture_arrays(capture):
    return {
        "raw": capture.raw,
        "ticks": capture.ticks,
        "seconds": capture.seconds,
        "dark": capture.dark,
        "white": capture.white,
        "metadata_utf8": np.frombuffer(
            canonical_json(capture.metadata.as_dict()), dtype=np.uint8
        ),
    }


def capture_bytes(capture):
    output = io.BytesIO()
    np.savez_compressed(output, **capture_arrays(capture))
    return output.getvalue()


def capture_hash(capture):
    digest = hashlib.sha256()
    for name, array in capture_arrays(capture).items():
        description = [name, array.dtype.str, list(array.shape)]
        digest.update(canonical_json(description))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()

