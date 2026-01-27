import os
import math
import time
import json
import requests
from tqdm import tqdm
from pyproj import Transformer, Geod
from shapely.geometry import box, Polygon, MultiPolygon
import geopandas as gpd
import osmnx as ox


def expand_bbox_for_osm(bbox, bbox_crs="EPSG:4326", tile_size_m=1024*0.25, n_tiles=2):
    """
    Expand bbox by n_tiles in EPSG:2180 meters, then transform back to WGS84 for OSM query
    bbox: (minx, miny, maxx, maxy) in bbox_crs
    tile_size_m: tile_size_px * resolution_m
    n_tiles: number of tiles to pad
    Returns expanded bbox in WGS84
    """
    # 1) Transform to EPSG:2180 (meters)
    if bbox_crs != "EPSG:2180":
        transformer_to_2180 = Transformer.from_crs(bbox_crs, "EPSG:2180", always_xy=True)
        minx, miny = transformer_to_2180.transform(bbox[0], bbox[1])
        maxx, maxy = transformer_to_2180.transform(bbox[2], bbox[3])
    else:
        minx, miny, maxx, maxy = bbox

    # 2) Expand by n_tiles
    expand_m = n_tiles * tile_size_m
    minx -= expand_m
    miny -= expand_m
    maxx += expand_m
    maxy += expand_m

    # 3) Transform back to WGS84
    transformer_to_wgs = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    minlon, minlat = transformer_to_wgs.transform(minx, miny)
    maxlon, maxlat = transformer_to_wgs.transform(maxx, maxy)

    return (minlon, minlat, maxlon, maxlat)


def fetch_buildings_osm_whole_bbox(
    bbox,
    bbox_crs="EPSG:4326",
    buffer_factor=1.02,
    verbose=True
):
    """
    Download OSM building footprints for the whole bbox (one Overpass/osmnx query or small fallback).
    Returns GeoDataFrame reprojected to EPSG:2180 (clipped precisely to bbox).
    bbox: (minx, miny, maxx, maxy) in bbox_crs
    """
    # 1. transform bbox to WGS84 for OSM queries
    if bbox_crs not in ("EPSG:4326", "epsg:4326", "4326"):
        transformer_to_wgs = Transformer.from_crs(bbox_crs, "EPSG:4326", always_xy=True)
        minlon, minlat = transformer_to_wgs.transform(bbox[0], bbox[1])
        maxlon, maxlat = transformer_to_wgs.transform(bbox[2], bbox[3])
    else:
        minlon, minlat, maxlon, maxlat = bbox

    # normalize
    if minlon > maxlon:
        minlon, maxlon = maxlon, minlon
    if minlat > maxlat:
        minlat, maxlat = maxlat, minlat

    north, south, east, west = maxlat, minlat, maxlon, minlon

    if verbose:
        print(f"Fetching OSM buildings for WGS84 bbox: N={north}, S={south}, E={east}, W={west}")

    tags = {"building": True}
    gdf = None

    # Try bbox-based request if available
    try:
        if hasattr(ox, "geometries_from_bbox"):
            if verbose: print("Using ox.geometries_from_bbox(...)")
            gdf = ox.geometries_from_bbox(north=north, south=south, east=east, west=west, tags=tags)
    except Exception as e:
        if verbose: print("geometries_from_bbox failed:", e)
        gdf = None

    # fallback to features_from_polygon or point/radius
    if gdf is None or gdf.empty:
        try:
            # create a polygon in WGS84 (shapely)
            poly_wgs = box(minlon, minlat, maxlon, maxlat)
            # try geometries_from_polygon if available
            if hasattr(ox, "geometries_from_polygon"):
                if verbose: print("Falling back to ox.geometries_from_polygon(...)")
                gdf = ox.geometries_from_polygon(poly_wgs, tags=tags)
            else:
                # fallback to center+radius
                center_lon = (minlon + maxlon) / 2.0
                center_lat = (minlat + maxlat) / 2.0
                geod = Geod(ellps="WGS84")
                _, _, d = geod.inv(center_lon, center_lat, maxlon, maxlat)
                radius_m = d * buffer_factor * math.sqrt(2)
                if hasattr(ox, "features_from_point"):
                    if verbose: print(f"Falling back to features_from_point radius={radius_m:.0f}m")
                    gdf = ox.features_from_point((center_lat, center_lon), tags=tags, dist=radius_m)
                elif hasattr(ox, "geometries_from_point"):
                    if verbose: print(f"Falling back to geometries_from_point radius={radius_m:.0f}m")
                    gdf = ox.geometries_from_point((center_lat, center_lon), tags=tags, dist=radius_m)
                else:
                    raise RuntimeError("No suitable osmnx API found for fallback.")
        except Exception as e:
            raise RuntimeError("OSMnx query for whole bbox failed: " + str(e))

    if gdf is None or gdf.empty:
        if verbose: print("No OSM buildings found for the given bbox.")
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:2180")

    # keep only polygons/multipolygons
    if "geometry" not in gdf:
        raise RuntimeError("OSM result has no geometry column.")
    gdf_polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf_polys.empty:
        if verbose: print("No polygonal building footprints in OSM result.")
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:2180")

    # ensure WGS84 CRS then reproject to EPSG:2180
    if gdf_polys.crs is None:
        gdf_polys = gdf_polys.set_crs("EPSG:4326")
    else:
        try:
            gdf_polys = gdf_polys.to_crs("EPSG:4326")
        except Exception:
            pass

    gdf_2180 = gdf_polys.to_crs("EPSG:2180")

    # Clip precisely to bbox in EPSG:2180
    transformer_to_2180 = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    minx2180, miny2180 = transformer_to_2180.transform(minlon, minlat)
    maxx2180, maxy2180 = transformer_to_2180.transform(maxlon, maxlat)
    bbox_poly_2180 = box(minx2180, miny2180, maxx2180, maxy2180)

    gdf_clipped = gdf_2180[gdf_2180.geometry.intersects(bbox_poly_2180)].copy()
    gdf_clipped.reset_index(drop=True, inplace=True)

    if verbose: print(f"Fetched {len(gdf_clipped)} building footprints (EPSG:2180).")
    return gdf_clipped


def download_tiles_with_cached_buildings(
    output_dir: str,
    metadata_path : str,
    bbox: tuple,
    buildings_gdf: gpd.GeoDataFrame = None,
    buildings_geojson_path: str | None = None,
    bbox_crs: str = "EPSG:4326",
    wcs_url: str = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WCS/StandardResolution",
    coverage_id: str | None = None,
    tile_size_px: int = 1024,
    resolution_m: float = 0.25,
    max_tiles: int | None = None,
    sleep_between_wcs: float = 0.2,
    timeout: int = 120,
    save_metadata_every: int = 100,
    verbose: bool = True,
    eps_m: float = 1.0,           # <-- small buffer (meters) to avoid edge-miss
    debug_show_empty_tile: bool = False  # show details when tile metadata is empty
):
    """
    Download tiles and attach building polygons (from cached buildings_gdf) per tile.
    This version expands the tile bbox by eps_m when querying the spatial index to
    avoid missing buildings that fall exactly on tile boundaries due to precision issues.
    """
    os.makedirs(output_dir, exist_ok=True)
    tile_size_m = tile_size_px * resolution_m

    # transform bbox to EPSG:2180 if necessary
    if bbox_crs != "EPSG:2180":
        transformer = Transformer.from_crs(bbox_crs, "EPSG:2180", always_xy=True)
        minx, miny = transformer.transform(bbox[0], bbox[1])
        maxx, maxy = transformer.transform(bbox[2], bbox[3])
    else:
        minx, miny, maxx, maxy = bbox

    if minx >= maxx or miny >= maxy:
        raise ValueError("Invalid bbox (min >= max).")

    # determine tile index range
    ix_min = math.floor(minx / tile_size_m)
    ix_max = math.ceil(maxx / tile_size_m) - 1
    iy_min = math.floor(miny / tile_size_m)
    iy_max = math.ceil(maxy / tile_size_m) - 1

    # load or fetch buildings if needed (same logic as before)
    if buildings_gdf is None:
        if verbose:
            print("Fetching buildings for whole bbox (this may take a while)...")
        expanded_bbox_wgs84 = expand_bbox_for_osm(bbox, bbox_crs=bbox_crs, tile_size_m=1024*0.25, n_tiles=1)
        buildings_gdf = fetch_buildings_osm_whole_bbox(expanded_bbox_wgs84, bbox_crs=bbox_crs, out_geojson=buildings_geojson_path, verbose=verbose)

    if buildings_gdf is None:
        buildings_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:2180")

    # build spatial index for fast queries
    try:
        sindex = buildings_gdf.sindex
    except Exception:
        sindex = None

    # try to auto-detect coverage_id
    if coverage_id is None:
        try:
            from owslib.wcs import WebCoverageService
            wcs = WebCoverageService(wcs_url, version="1.0.0")
            keys = list(wcs.contents.keys())
            if keys:
                coverage_id = keys[0]
                if verbose: print("Auto-selected coverage_id:", coverage_id)
        except Exception as e:
            if verbose: print("Warning: failed to auto-detect coverage_id:", e)

    metadata = {}
    saved_tiles = 0
    tile_list = []
    for ix in range(ix_min, ix_max + 1):
        for iy in range(iy_min, iy_max + 1):
            tile_minx = ix * tile_size_m
            tile_miny = iy * tile_size_m
            tile_maxx = (ix + 1) * tile_size_m
            tile_maxy = (iy + 1) * tile_size_m
            if tile_maxx <= minx or tile_minx >= maxx or tile_maxy <= miny or tile_miny >= maxy:
                continue
            tile_list.append((ix, iy, tile_minx, tile_miny, tile_maxx, tile_maxy))

    if verbose: print(f"Total tiles to process: {len(tile_list)}")

    for idx, (ix, iy, tile_minx, tile_miny, tile_maxx, tile_maxy) in enumerate(tqdm(tile_list, desc="tiles")):
        if max_tiles is not None and saved_tiles >= max_tiles:
            if verbose: print("Reached max_tiles. Stopping.")
            break

        img_id = f"x{ix}_y{iy}"
        fname = os.path.join(output_dir, f"{img_id}.tif")

        # Download tile if missing (same as before) ...
        if not os.path.exists(fname):
            params = {
                "SERVICE": "WCS",
                "VERSION": "1.0.0",
                "REQUEST": "GetCoverage",
                "COVERAGE": coverage_id if coverage_id else "",
                "IDENTIFIER": coverage_id if coverage_id else "",
                "CRS": "EPSG:2180",
                "BBOX": f"{tile_minx},{tile_miny},{tile_maxx},{tile_maxy}",
                "FORMAT": "GeoTIFF",
                "WIDTH": str(tile_size_px),
                "HEIGHT": str(tile_size_px),
            }
            params = {k: v for k, v in params.items() if v != ""}
            try:
                r = requests.get(wcs_url, params=params, stream=True, timeout=timeout)
            except Exception as e:
                if verbose: print(f"WCS request failed for {img_id}: {e}")
                continue

            if r.status_code != 200:
                snippet = r.content.decode("utf-8", errors="replace")[:800]
                if verbose: print(f"WCS HTTP {r.status_code} for {img_id}. Snippet:\n{snippet}")
                continue

            try:
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                if verbose: print("Failed writing tile:", e)
                if os.path.exists(fname):
                    os.remove(fname)
                continue
            time.sleep(sleep_between_wcs)

        # Clip buildings for this tile using spatial index with small epsilon
        tile_meta_list = []
        if not buildings_gdf.empty:
            tile_bbox_geom = box(tile_minx, tile_miny, tile_maxx, tile_maxy)
            # Expand bbox slightly when querying sindex to catch edge features
            minx_q = tile_minx - eps_m
            miny_q = tile_miny - eps_m
            maxx_q = tile_maxx + eps_m
            maxy_q = tile_maxy + eps_m

            if sindex:
                candidate_idx = list(sindex.intersection((minx_q, miny_q, maxx_q, maxy_q)))
            else:
                candidate_idx = list(range(len(buildings_gdf)))

            if candidate_idx:
                candidates = buildings_gdf.iloc[candidate_idx]
                # precise intersection with a slightly buffered tile bbox
                inter = candidates[candidates.geometry.intersects(tile_bbox_geom.buffer(eps_m))].copy()
            else:
                inter = gpd.GeoDataFrame(columns=buildings_gdf.columns, crs=buildings_gdf.crs)

            # If debugging and we find zero intersecting geometries but the tile image shows buildings,
            # print a hint so you can investigate that tile.
            if debug_show_empty_tile and inter.empty:
                if verbose:
                    print(f"DEBUG: tile {img_id} has 0 matched buildings after EPS={eps_m}m expansion.")
                    # show how many candidates were in index
                    if sindex:
                        print(f"  candidates in sindex search: {len(candidate_idx)}")
                    # optionally list bounding boxes of candidates (first few)
                    if len(candidate_idx) > 0 and verbose:
                        sample = buildings_gdf.iloc[candidate_idx[:5]]
                        print("  sample candidate bboxes:")
                        for ii, g in sample.geometry.bounds.iteritems():
                            pass  # no-op (kept for possible extension)

            # convert each polygon to tile-local pixel coordinates
            for geom in inter.geometry:
                polys = []
                if isinstance(geom, MultiPolygon):
                    polys = list(geom.geoms)
                elif isinstance(geom, Polygon):
                    polys = [geom]
                else:
                    continue

                for poly in polys:
                    exterior = poly.exterior.coords[:]
                    px_coords = []
                    for (x, y) in exterior:
                        px = (x - tile_minx) / resolution_m
                        py = tile_size_px - (y - tile_miny) / resolution_m
                        px = max(0.0, min(px, float(tile_size_px)))
                        py = max(0.0, min(py, float(tile_size_px)))
                        px_coords.append([round(px, 2), round(py, 2)])

                    xs = [p[0] for p in px_coords]
                    ys = [p[1] for p in px_coords]
                    x_min_px = int(math.floor(min(xs)))
                    y_min_px = int(math.floor(min(ys)))
                    x_max_px = int(math.ceil(max(xs)))
                    y_max_px = int(math.ceil(max(ys)))
                    width_px = max(0, x_max_px - x_min_px)
                    height_px = max(0, y_max_px - y_min_px)

                    tile_meta_list.append({
                        "polygon": px_coords,
                        "box": [x_min_px, y_min_px, width_px, height_px]
                    })

        metadata[img_id] = tile_meta_list
        saved_tiles += 1

        # periodic metadata save
        if (saved_tiles % save_metadata_every) == 0:
            meta_path = os.path.join(metadata_path)
            try:
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(metadata, mf, ensure_ascii=False, indent=2)
                if verbose: print(f"Saved metadata to {meta_path} after {saved_tiles} tiles.")
            except Exception as e:
                if verbose: print("Failed to save metadata interim:", e)

    # final metadata write
    meta_path = os.path.join(metadata_path)
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, ensure_ascii=False, indent=2)
        if verbose: print(f"Final metadata saved to {meta_path}. Total tiles: {saved_tiles}")
    except Exception as e:
        if verbose: print("Failed to save final metadata:", e)

    return metadata


# Download
bbox_wgs84 = (16.7315675, 52.2919282, 17.071629, 52.5092936)
expanded_bbox_wgs84 = expand_bbox_for_osm(bbox_wgs84, bbox_crs="EPSG:4326", tile_size_m=1024*0.25, n_tiles=1)
buildings = fetch_buildings_osm_whole_bbox(expanded_bbox_wgs84, bbox_crs="EPSG:4326")

meta = download_tiles_with_cached_buildings(
    output_dir="imgs",
    metadata_path="metadata.json",
    bbox=bbox_wgs84,
    buildings_gdf=buildings,
    bbox_crs="EPSG:4326",
    coverage_id=None,
    max_tiles=5000,
    save_metadata_every=100
)
