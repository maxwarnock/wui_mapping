'''*********************************************
WUI Mapping - Data Download Script
Downloads and stages NLCD land cover and HISDAC-US BUPL data for a
given state, ready to be used by wui_mapping_tool.py.

This script is intentionally kept separate from wui_mapping_tool.py /
wui_mapping_module.py: it needs geopandas/rasterio, which can conflict
with the pinned package versions ArcGIS Pro's arcpy environment relies
on. Run this script in its own environment (see README below), then
run wui_mapping_tool.py in the arcpy environment as usual, pointing it
at the files this script produces.

NLCD is fetched via a direct WMS GetMap request rather than through
the pygeohydro package. pygeohydro's nlcd_bygeom() goes through
pygeoutils.gtiff2xarray(), which was found (by isolating the crash
step by step) to return a lazily-loaded rioxarray DataArray backed by
a rasterio MemoryFile that gets closed before the pixel data is ever
read - a use-after-free that segfaults the process nondeterministically
rather than raising a catchable Python exception. Fetching tiles
directly and reading them into a plain numpy array while the
MemoryFile is still open (proven safe here) avoids that code path
entirely.

BUPL is distributed as a single CONUS-wide raster (~214 million cells).
Feeding that directly to arcpy's sa.ExtractByRectangle() alongside a
state-sized raster was found to reliably crash ArcGIS Pro's Python
(STATUS_HEAP_CORRUPTION, 0xC0000374 - reproduced with a minimal script
and confirmed to match the fault signature of the real crash). BUPL is
now clipped to the target state's bounding box (with a buffer) here,
before wui_mapping_tool.py ever sees it, which avoids that code path.
The clip also avoids rasterio.windows.from_bounds(), which - like
rasterio.merge.merge() - was independently found to hard-crash this
environment's rasterio 1.4.4 build; the window is computed with plain
arithmetic on the affine transform instead (verified safe).

Setup (one time):
    conda create -n wui_data_download -c conda-forge python=3.11 geopandas rasterio requests -y
    conda activate wui_data_download

Usage:
    python download_data.py
        (prompts interactively for which state and year to download)
    python download_data.py --state CO --year 2020
        (non-interactive, for scripting)
*********************************************'''

import argparse
import io
import math
import os
import shutil
import tarfile
import zipfile

import geopandas as gpd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CENSUS_STATE_BOUNDARY_URL = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"

# Historical built-up property locations (BUPL) Version II, 1810-2020, HISDAC-US
# Harvard Dataverse: https://doi.org/10.7910/DVN/U2P66Z
HISDAC_BUPL_FILE_ID = "8961704"
HISDAC_BUPL_DOWNLOAD_URL = f"https://dataverse.harvard.edu/api/access/datafile/{HISDAC_BUPL_FILE_ID}"

# NLCD land cover is only published for specific survey years, not annually.
NLCD_COVER_YEARS = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]

NLCD_WMS_URL = "https://www.mrlc.gov/geoserver/mrlc_download/wms"

# Rough approximation (meters per degree of latitude); longitude is corrected
# by cos(latitude). Only used to size tile requests in pixels - the final
# raster gets resampled to BUPL's grid downstream anyway, so this doesn't
# need to be geodetically precise.
METERS_PER_DEGREE_LAT = 111_320

# Keep each individual WMS request comfortably small (fast, low memory,
# unlikely to hit server-side limits) - tiles are mosaicked locally afterward.
MAX_PIXELS_PER_TILE = 4_000_000


def load_state_boundaries():
    '''Downloads (or reuses a cached copy of) the Census TIGER/Line state
    boundary shapefile and returns it as a GeoDataFrame in EPSG:4326.'''
    extract_dir = os.path.join(DATA_DIR, "_tiger_states")
    shp_candidates = [f for f in os.listdir(extract_dir) if f.endswith(".shp")] if os.path.isdir(extract_dir) else []

    if not shp_candidates:
        print("Downloading Census TIGER state boundaries...")
        resp = requests.get(CENSUS_STATE_BOUNDARY_URL, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            os.makedirs(extract_dir, exist_ok=True)
            zf.extractall(extract_dir)
        shp_candidates = [f for f in os.listdir(extract_dir) if f.endswith(".shp")]

    shp_path = os.path.join(extract_dir, shp_candidates[0])
    return gpd.read_file(shp_path).to_crs("EPSG:4326")


def select_state(states, query):
    '''Matches query against state abbreviation (e.g. "CO") or full name
    (e.g. "Colorado"), case-insensitive. Returns a single-row GeoDataFrame,
    or None if no match is found.'''
    query = query.strip()
    match = states[states["STUSPS"].str.lower() == query.lower()]
    if match.empty:
        match = states[states["NAME"].str.lower() == query.lower()]
    return None if match.empty else match


def prompt_state(states):
    '''Prompts until the user enters a valid state abbreviation or name.'''
    while True:
        query = input("Which state would you like to download? (name or 2-letter abbreviation): ")
        match = select_state(states, query)
        if match is not None:
            print(f"Selected {match.iloc[0]['NAME']} ({match.iloc[0]['STUSPS']})")
            return match
        print(f"'{query}' did not match a US state. Try a name like 'Colorado' or abbreviation like 'CO'.")


def prompt_year(default=2020):
    '''Prompts until the user enters a valid year, or accepts the default on
    a blank answer. This is approximate for NLCD: land cover is only
    published for specific survey years (see NLCD_COVER_YEARS), so whatever
    is entered gets snapped to the nearest one available - download_nlcd()
    reports when it does.'''
    while True:
        raw = input(f"Which year would you like to download data for (approximate; press Enter for {default}): ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("Please enter a whole number year, e.g. 2020.")


def _split_bbox(bounds, resolution):
    '''Splits bounds (minx, miny, maxx, maxy in EPSG:4326 degrees) into a
    grid of smaller bboxes, sized so each one stays under
    MAX_PIXELS_PER_TILE at the given resolution (meters). Returns a list
    of (minx, miny, maxx, maxy) tuples.'''
    minx, miny, maxx, maxy = bounds
    mean_lat = (miny + maxy) / 2
    width_m = (maxx - minx) * METERS_PER_DEGREE_LAT * math.cos(math.radians(mean_lat))
    height_m = (maxy - miny) * METERS_PER_DEGREE_LAT
    total_pixels = (width_m / resolution) * (height_m / resolution)
    n_tiles = max(1, math.ceil(total_pixels / MAX_PIXELS_PER_TILE))
    n_side = math.ceil(math.sqrt(n_tiles))

    dx = (maxx - minx) / n_side
    dy = (maxy - miny) / n_side
    return [
        (minx + i * dx, miny + j * dy, minx + (i + 1) * dx, miny + (j + 1) * dy)
        for i in range(n_side)
        for j in range(n_side)
    ]


def _fetch_nlcd_tile(bbox, year, deg_per_px_lon, deg_per_px_lat):
    '''Fetches one NLCD land cover tile via a direct WMS GetMap request
    and reads it into a plain numpy array. Returns (array, transform, crs, profile).

    deg_per_px_lon/lat must be the same for every tile in a mosaic (computed
    once for the whole area by download_nlcd) - using a per-tile latitude to
    derive these, instead of one shared value, produces tiles with slightly
    different pixel widths that don't share a common grid.

    This deliberately bypasses pygeohydro.nlcd_bygeom(), which goes
    through pygeoutils.gtiff2xarray() - found (by isolating the crash
    step by step) to return a lazily-loaded rioxarray DataArray backed
    by a rasterio MemoryFile that gets closed before the pixel data is
    ever read, a use-after-free that segfaults nondeterministically
    rather than raising a catchable Python exception. Reading the array
    eagerly while the MemoryFile is still open (as done here) avoids
    that code path entirely - verified safe through direct testing.'''
    import rasterio as rio

    minx, miny, maxx, maxy = bbox
    width = max(1, round((maxx - minx) / deg_per_px_lon))
    height = max(1, round((maxy - miny) / deg_per_px_lat))

    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": f"NLCD_{year}_Land_Cover_L48",
        "styles": "",
        "crs": "CRS:84",
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "width": str(width),
        "height": str(height),
        "format": "image/geotiff",
    }
    resp = requests.get(NLCD_WMS_URL, params=params, timeout=120)
    resp.raise_for_status()

    with rio.MemoryFile(resp.content) as memfile:
        with memfile.open() as src:
            array = src.read()
            transform = src.transform
            crs = src.crs
            profile = src.profile
    return array, transform, crs, profile


def _mosaic_tiles(tile_paths, out_path):
    '''Mosaics a list of non-overlapping, same-resolution GeoTIFF tiles
    into out_path by placing each tile's array directly into a
    pre-allocated numpy array at its geographic offset.

    This deliberately avoids rasterio.merge.merge(), which was found
    (by isolating the crash on minimal synthetic tiles, unrelated to
    NLCD or the network) to hard-crash the process in this environment
    (rasterio 1.4.4) regardless of tile alignment or content - a bug in
    that function's own native code, not in the tiling logic here.'''
    import rasterio
    import numpy as np
    from rasterio.transform import from_origin

    srcs_meta = []
    for p in tile_paths:
        with rasterio.open(p) as src:
            srcs_meta.append((p, src.bounds, src.transform, src.crs, src.dtypes[0], src.count))

    minx = min(b.left for _, b, *_ in srcs_meta)
    maxx = max(b.right for _, b, *_ in srcs_meta)
    miny = min(b.bottom for _, b, *_ in srcs_meta)
    maxy = max(b.top for _, b, *_ in srcs_meta)

    _, _, transform0, crs0, dtype0, count0 = srcs_meta[0]
    px_w = transform0.a
    px_h = -transform0.e

    out_width = round((maxx - minx) / px_w)
    out_height = round((maxy - miny) / px_h)
    out_transform = from_origin(minx, maxy, px_w, px_h)

    mosaic = np.zeros((count0, out_height, out_width), dtype=dtype0)
    for p, bounds, _transform, _crs, _dtype, _count in srcs_meta:
        with rasterio.open(p) as src:
            arr = src.read()
        col_off = round((bounds.left - minx) / px_w)
        row_off = round((maxy - bounds.top) / px_h)
        h, w = arr.shape[1], arr.shape[2]
        mosaic[:, row_off:row_off + h, col_off:col_off + w] = arr

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=out_height, width=out_width,
        count=count0, dtype=dtype0, crs=crs0, transform=out_transform,
    ) as dst:
        dst.write(mosaic)


# HISDAC-US BUPL's CRS - "USA_Contiguous_Albers_Equal_Area_Conic_USGS_version".
# The working LA sample data that shipped with this repo was already in this
# CRS (as raw WKT, not this ESRI code, but numerically identical: same standard
# parallels/center). NLCD fetched via the WMS above comes back in EPSG:4326
# (geographic degrees) and untiled/striped, unlike that sample. Feeding
# wui_mapping_tool.py a geographic-CRS, untiled raster and making its
# ProjectRaster call do a real geographic-to-projected transform (instead of
# the near-no-op the pre-reprojected sample required) is suspected to be why
# ProjectRaster crashes intermittently there but never in isolated testing
# here - see conversation history. Reprojecting NLCD to this CRS and tiling it
# here, matching the sample's format, avoids relying on that ArcGIS code path.
NLCD_TARGET_CRS = "ESRI:102039"


def _reproject_to_crs(src_path, dst_path, dst_crs):
    '''Reprojects src_path to dst_crs and writes a tiled GeoTIFF to dst_path,
    matching the format of the original working sample data (see
    NLCD_TARGET_CRS docstring above). Verified safe in isolated testing,
    unlike rasterio.merge.merge()/windows.from_bounds() elsewhere in this file.'''
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs, transform=transform, width=width, height=height,
            driver="GTiff", tiled=True, blockxsize=128, blockysize=128,
        )
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_i),
                    destination=rasterio.band(dst, band_i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )


def download_nlcd(state_geom, year, out_path, resolution=30):
    '''Downloads NLCD land cover clipped to state_geom's bounding box, at
    the given resolution (meters), and writes it to out_path. NLCD land
    cover is only published for specific survey years (see
    NLCD_COVER_YEARS), so this snaps the requested year to the nearest
    available one and returns the year actually used.

    Note: this clips to the state's bounding box, not its exact polygon
    boundary, so the output includes small slivers of neighboring states
    near the corners. wui_mapping_tool.py resamples this raster down to
    BUPL's ~250m cell size anyway, so the coarser resolution used here
    loses nothing downstream.'''
    actual_year = min(NLCD_COVER_YEARS, key=lambda y: abs(y - year))
    if actual_year != year:
        print(f"NLCD land cover isn't published for {year}; using the nearest available year, {actual_year}, instead.")

    bounds = state_geom.geometry.unary_union.bounds
    minx, miny, maxx, maxy = bounds
    mean_lat = (miny + maxy) / 2
    deg_per_px_lat = resolution / METERS_PER_DEGREE_LAT
    deg_per_px_lon = resolution / (METERS_PER_DEGREE_LAT * math.cos(math.radians(mean_lat)))

    tiles = _split_bbox(bounds, resolution)
    print(f"Requesting NLCD {actual_year} land cover at {resolution}m resolution, "
          f"split into {len(tiles)} tile(s) via direct WMS calls (this can take a while)...")

    tile_dir = os.path.join(DATA_DIR, "_nlcd_tiles")
    os.makedirs(tile_dir, exist_ok=True)
    tile_paths = []
    try:
        import rasterio

        for i, bbox in enumerate(tiles):
            print(f"  Downloading tile {i + 1}/{len(tiles)}...")
            array, transform, crs, profile = _fetch_nlcd_tile(bbox, actual_year, deg_per_px_lon, deg_per_px_lat)
            tile_path = os.path.join(tile_dir, f"tile_{i}.tif")
            profile.update(driver="GTiff")
            with rasterio.open(tile_path, "w", **profile) as dst:
                dst.write(array)
            tile_paths.append(tile_path)

        mosaic_path = os.path.join(tile_dir, "_mosaic_4326.tif")
        print("Mosaicking tiles...")
        _mosaic_tiles(tile_paths, mosaic_path)

        print(f"Reprojecting to {NLCD_TARGET_CRS} (matching BUPL's CRS)...")
        _reproject_to_crs(mosaic_path, out_path, NLCD_TARGET_CRS)
    finally:
        shutil.rmtree(tile_dir, ignore_errors=True)

    print(f"Saved NLCD {actual_year} land cover to {out_path}")
    return actual_year


def _clip_raster_to_bounds(src_path, bounds_4326, out_path, buffer_deg=0.5):
    '''Clips src_path (any CRS) down to bounds_4326 (minx, miny, maxx, maxy
    in EPSG:4326), padded by buffer_deg, using a windowed read - only the
    needed portion of the source file is ever touched, so this is fast and
    memory-light even for a huge source raster.

    Deliberately avoids rasterio.windows.from_bounds(), which - like
    rasterio.merge.merge() - was found to hard-crash this environment's
    rasterio 1.4.4 build. The pixel window is computed with plain
    arithmetic on the affine transform instead (verified safe).'''
    import rasterio
    from rasterio.windows import Window
    from pyproj import Transformer

    minx, miny, maxx, maxy = bounds_4326
    minx, miny, maxx, maxy = minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg

    with rasterio.open(src_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
        proj_minx, proj_maxx = min(xs), max(xs)
        proj_miny, proj_maxy = min(ys), max(ys)

        t = src.transform
        col_off = (proj_minx - t.c) / t.a
        row_off = (t.f - proj_maxy) / (-t.e)
        col_stop = (proj_maxx - t.c) / t.a
        row_stop = (t.f - proj_miny) / (-t.e)

        col_off_i = max(0, int(col_off))
        row_off_i = max(0, int(row_off))
        col_stop_i = min(src.width, int(col_stop) + 1)
        row_stop_i = min(src.height, int(row_stop) + 1)
        window = Window(col_off_i, row_off_i, col_stop_i - col_off_i, row_stop_i - row_off_i)

        data = src.read(window=window)
        out_transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(height=window.height, width=window.width, transform=out_transform, driver="GTiff")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


def download_bupl(year, out_path, bounds_4326=None):
    '''Downloads the HISDAC-US BUPL archive (all years, CONUS extent),
    extracts it, and saves the requested year's raster to out_path. If
    bounds_4326 (minx, miny, maxx, maxy in EPSG:4326) is given, the
    output is clipped to it (with a buffer) rather than left at the
    full CONUS extent - see _clip_raster_to_bounds and the module
    docstring for why this matters (avoids crashing wui_mapping_tool.py).'''
    print("Downloading HISDAC-US BUPL archive (~640 MB, all years, CONUS extent)...")
    tar_path = os.path.join(DATA_DIR, "_BUPL.tar")
    os.makedirs(DATA_DIR, exist_ok=True)
    with requests.get(HISDAC_BUPL_DOWNLOAD_URL, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(tar_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    print("Extracting BUPL archive...")
    extract_dir = os.path.join(DATA_DIR, "_hisdac_bupl")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_dir)

    year_files = [
        os.path.join(root, f)
        for root, _dirs, files in os.walk(extract_dir)
        for f in files
        if str(year) in f and f.lower().endswith((".tif", ".tiff"))
    ]
    if not year_files:
        raise FileNotFoundError(
            f"Could not find a BUPL raster for year {year} inside the extracted archive. "
            f"Check {extract_dir} and adjust the matching logic above."
        )
    if len(year_files) > 1:
        print(f"Warning: multiple candidate files matched year {year}, using the first: {year_files}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if bounds_4326 is not None:
        print("Clipping BUPL to the study area...")
        _clip_raster_to_bounds(year_files[0], bounds_4326, out_path)
    else:
        shutil.copy(year_files[0], out_path)
    print(f"Saved BUPL {year} to {out_path}")

    os.remove(tar_path)
    shutil.rmtree(extract_dir)


def main():
    parser = argparse.ArgumentParser(description="Download NLCD + HISDAC-US BUPL data for a state")
    parser.add_argument("--state", help="State name or two-letter abbreviation, e.g. Colorado or CO. "
                                         "If omitted, you'll be prompted for it interactively.")
    parser.add_argument("--year", type=int, help="Year to download for both datasets. "
                                                  "If omitted, you'll be prompted for it interactively (default: 2020).")
    parser.add_argument("--resolution", type=int, default=90,
                         help="NLCD resolution in meters (default: 90). NLCD's native resolution is 30m, "
                              "but requesting a whole state at 30m has been observed to crash the download "
                              "(out-of-memory). Since wui_mapping_tool.py resamples this down to BUPL's ~250m "
                              "cell size anyway, 90m is a safe default; drop to 30 only for small areas.")
    parser.add_argument("--skip-nlcd", action="store_true", help="Skip the NLCD download step")
    parser.add_argument("--skip-bupl", action="store_true", help="Skip the BUPL download step")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    year = args.year if args.year is not None else prompt_year()

    nlcd_year = None
    bupl_year = None

    # State is needed for both steps now: it defines the NLCD request area,
    # and BUPL gets clipped to it too (see download_bupl/_clip_raster_to_bounds
    # docstrings - feeding wui_mapping_tool.py the full CONUS-wide BUPL raster
    # was found to crash ArcGIS Pro).
    state_geom = None
    if not args.skip_nlcd or not args.skip_bupl:
        states = load_state_boundaries()
        if args.state:
            state_geom = select_state(states, args.state)
            if state_geom is None:
                raise ValueError(f"'{args.state}' did not match a US state name or abbreviation")
            print(f"Selected {state_geom.iloc[0]['NAME']} ({state_geom.iloc[0]['STUSPS']})")
        else:
            state_geom = prompt_state(states)

    state_abbr = state_geom.iloc[0]["STUSPS"] if state_geom is not None else None

    if not args.skip_nlcd:
        # Output filename reflects whatever year NLCD actually used, which
        # may differ from the requested year since NLCD land cover isn't annual.
        nlcd_year = min(NLCD_COVER_YEARS, key=lambda y: abs(y - year))
        nlcd_out = os.path.join(DATA_DIR, f"nlcd_{state_abbr}_{nlcd_year}.tif")
        download_nlcd(state_geom, year, nlcd_out, resolution=args.resolution)

    if not args.skip_bupl:
        bupl_year = year
        bupl_out = os.path.join(DATA_DIR, f"BUPL_{state_abbr}_{bupl_year}.tif")
        bounds_4326 = state_geom.geometry.unary_union.bounds if state_geom is not None else None
        download_bupl(bupl_year, bupl_out, bounds_4326=bounds_4326)

    print("Done.")
    if nlcd_year is not None:
        print(f"  NLCD file: data/nlcd_{state_abbr}_{nlcd_year}.tif")
    if bupl_year is not None:
        print(f"  BUPL file: data/BUPL_{state_abbr}_{bupl_year}.tif")
    if nlcd_year is not None and bupl_year is not None and nlcd_year != bupl_year:
        print(f"  Note: NLCD ({nlcd_year}) and BUPL ({bupl_year}) are different years since NLCD land cover isn't published annually.")
    print("wui_mapping_tool.py will pick these up automatically (it finds the most recently downloaded files).")


if __name__ == "__main__":
    main()
