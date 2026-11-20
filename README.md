# RollScope

RollScope is a local inspection workbench for encoder-indexed line-scan captures of rolled material. It keeps scan order, encoder station, and detector column separate. Review starts with the recorded sensor codes and ends with physical measurements, a saved inspection, or a portable evidence archive.

## Run

```sh
docker compose up -d --build
```

Open http://127.0.0.1:18945/. The first start creates a clearly labeled synthetic cold-rolled steel capture. The capture library also provides a synthetic coated-aluminum example. Compose listens only on the loopback address and keeps its SQLite database in the `solo45-rollscope_capture-data` volume. `docker compose down` stops this copy while retaining the volume. Set `ROLLSCOPE_PORT` to choose a different free loopback port.

For a native installation, use Python 3.11 or 3.12 with the pinned packages in `requirements.txt`.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Use `ROLLSCOPE_DATA_DIR` for the data directory and `PORT` for the native server port. When the packages are already available through the interpreter's search path, start with `python3 app.py`. Use `python3 -m unittest discover -s tests -v` to run the product tests in that environment.

## Inspect a capture

- **Raw scan** displays sensor codes, clock pauses, encoder jumps, repeated ticks and independent transport speed estimates.
- **Flat field** corrects reflectance with the median dark and white references. It flags weak response, unstable columns and saturation. Its reference-pass plots distinguish localized variation from a whole reference sweep.
- **Station view** puts lines on a uniform longitudinal grid. An exact encoder tick uses the mean of usable repeat scans at that tick. A station between ticks uses a linear blend with two usable neighbors. Wide encoder gaps stay missing.
- **Coverage** displays the distinct native, blended, absent and sensor-invalid categories and lists the bracketing raw lines for acquisition gaps.
- **Strip geometry** follows the supported material span, with a trace for each observed threshold crossing and millimetre width and center measurements. A clipped edge gives an unavailable width.
- **Surface spots** compares pixels with a local background inside the measured strip. It keeps bright and dark footprints and identifies regions that touch unavailable evidence.
- **Streaks** tracks persistent dark or bright contrast along the material without joining unsupported stretches of a detector column.
- **Repeat pitch** finds dark marks in a cross-strip band and measures the observed spacing. An interval crossing unsupported evidence is marked unknown.
- **Transverse joins** records rows whose same-sign contrast extends across the required fraction of measured material.
- **Texture** derives stripe orientation and wavelength bands from complete, detrended, windowed tiles in physical units.
- **Reflectance trend** summarizes along-roll and cross-strip profiles and a histogram in the valid material interior.
- **Probe** selects a raster cell and shows its physical coordinates, raw codes and effective source weights. The transverse profile comes from the same selected image row.
- **Two-level framing** keeps station, coverage and spot views on the same physical frame. When a full roll is squeezed into one preview, each thumbnail pixel is a block of source cells: its colour is the block median so bright/dark trends survive, while narrow missing lines, invalid samples and threshold-crossing anomalies that occupy only part of a block are kept as distinct partial markers instead of being averaged away or expanded to fill the cell. Selecting a block reports its actual station range, transverse range, support counts and the evidence that occupies only part of it, and offers a zoom into that exact frame; the zoomed view probes individual source cells. A saved review window is also drawn as a clickable frame and opens the same zoom entry.
- **Review windows** store physical bounds, status and a note. Window evidence includes valid grid cells, contributing raw samples, profiles, source weights, the marked PNG and a review sheet.
- **Ruler** measures a two-point path in millimetres and partitions its length into usable material, background, bad pixels and encoder gaps. The path archive contains the plotted image and sampled-cell table.
- **Saved reports** retain a result and the settings that produced it. Compare the saved summary with today's settings, or export the saved report and its CSV measurements.
- **Capture library** downloads the original-equivalent NPZ, imports a supported NPZ and backs up or restores the whole capture with saved notes, windows and reports.

Open **Capture format** for the NPZ shapes and physical units. This input contains a monochrome raster of 8–4096 lines by 16–1024 detector columns, nondecreasing integer encoder ticks, increasing times and independent dark and white reference passes. NPZ downloads preserve the numeric samples and geometry. For uploads the app records both the original file checksum and the canonical capture checksum. Evidence archives include per-member checksums. A project restore creates a new library entry and retains the source capture checksum.

## Measurement limits

The supplied captures are generated from deterministic patterns. They do not represent mill observations. Analyses are exploratory and require review of the source images. The current coordinate model covers one forward capture on a fixed detector geometry. Station values are relative to the encoded origin. The display stretches the raster to fill its viewport, while axes and measurements use millimetres. Preview rendering may subsample large rasters. Evidence NPZ arrays retain the selected grid resolution.

The library supports up to 64 captures and 512 MiB of stored sources, with 40 windows and 40 saved reports per capture. A station grid is capped at 3,000,000 cells. Capture uploads and project restores accept files up to 48 MiB. A screenshot, clipped track or candidate count does not establish the true extent of an anomaly.

Full-roll previews render at most 1,000 by 1,400 blocks. A block median represents the broad reflectance trend and is never substituted for the source values; the block inspector reports the exact station and transverse bounds plus every partial evidence category, and zooming the frame restores one image pixel per source cell. Ad-hoc zoom frames are transient request bounds and are not stored as review windows.
