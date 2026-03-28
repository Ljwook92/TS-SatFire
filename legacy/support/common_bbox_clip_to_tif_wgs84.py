import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.warp import Resampling, reproject


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clip GOES FDCF NetCDF files, then save mask/frp on each event's "
            "TS-SatFire VIIRS_Day WGS84 grid."
        )
    )
    parser.add_argument("--goes-root", required=True, help="Root directory of GOES FDCF NetCDF files.")
    parser.add_argument(
        "--bbox-csvs",
        nargs="+",
        required=True,
        help="List of bbox CSV paths, e.g. us_fire_2019_bbox.csv us_fire_2020_bbox.csv us_fire_2021_bbox.csv",
    )
    parser.add_argument(
        "--ts-root",
        default="/home/jlc3q/data/SatFire/ts-satfire",
        help="Root directory of TS-SatFire per-event folders.",
    )
    parser.add_argument("--output-root", required=True, help="Output root directory.")
    parser.add_argument("--summary-csv", required=True, help="Path to write the summary CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--limit-events", type=int, default=0, help="Optional limit on total events.")
    return parser.parse_args()


def daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def infer_year_from_csv_path(csv_path):
    name = Path(csv_path).stem
    for part in name.split("_"):
        if part.isdigit() and len(part) == 4:
            return part
    raise ValueError(f"Could not infer year from bbox CSV path: {csv_path}")


def load_rows(csv_paths):
    rows = []
    for csv_path in csv_paths:
        year = infer_year_from_csv_path(csv_path)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                row["_year"] = year
                rows.append(row)
    return rows


def build_goes_crs(ds):
    proj = ds["goes_imager_projection"]
    return CRS.from_proj4(
        " ".join(
            [
                "+proj=geos",
                f"+h={proj.perspective_point_height.item()}",
                f"+lon_0={proj.longitude_of_projection_origin.item()}",
                f"+sweep={proj.sweep_angle_axis}",
                "+ellps=GRS80",
                "+no_defs",
            ]
        )
    )


def compute_xy_bounds(ds, row):
    min_lon = float(row["min_lon"])
    min_lat = float(row["min_lat"])
    max_lon = float(row["max_lon"])
    max_lat = float(row["max_lat"])

    proj = ds["goes_imager_projection"]
    sat_height = proj.perspective_point_height.item()
    goes_crs = build_goes_crs(ds)
    transformer = Transformer.from_crs("EPSG:4326", goes_crs, always_xy=True)

    xs_m, ys_m = transformer.transform(
        [min_lon, min_lon, max_lon, max_lon],
        [min_lat, max_lat, min_lat, max_lat],
    )
    xs_rad = [x / sat_height for x in xs_m]
    ys_rad = [y / sat_height for y in ys_m]
    return min(xs_rad), max(xs_rad), min(ys_rad), max(ys_rad)


def clip_dataset(ds, xy_bounds):
    x_min, x_max, y_min, y_max = xy_bounds
    x_slice = slice(x_min, x_max)
    if ds.y.values[0] > ds.y.values[-1]:
        y_slice = slice(y_max, y_min)
    else:
        y_slice = slice(y_min, y_max)
    return ds.sel(x=x_slice, y=y_slice)


def build_source_transform(clipped):
    x = clipped.x.values
    y = clipped.y.values
    if x.size < 2 or y.size < 2:
        raise ValueError("Need at least two x/y coordinates to build source transform.")

    sat_height = clipped["goes_imager_projection"].perspective_point_height.item()
    x_m = x * sat_height
    y_m = y * sat_height

    x_res = float(x_m[1] - x_m[0])
    y_res = float(y_m[1] - y_m[0])
    x_origin = float(x_m[0] - x_res / 2.0)
    y_origin = float(y_m[0] - y_res / 2.0)
    return rasterio.Affine(x_res, 0.0, x_origin, 0.0, y_res, y_origin)


def list_event_files(goes_root, start_date, end_date):
    files = []
    for current_date in daterange(start_date, end_date):
        year = f"{current_date.year:04d}"
        doy = f"{current_date.timetuple().tm_yday:03d}"
        day_dir = Path(goes_root) / year / doy
        if day_dir.exists():
            files.extend(sorted(day_dir.rglob("*.nc")))
    return files


def extract_timestamp(nc_path, clipped):
    time_str = clipped.attrs.get("time_coverage_start", "")
    if time_str:
        return time_str.replace("-", "").replace(":", "").split(".")[0]
    stem = Path(nc_path).stem
    if "_s" in stem:
        return stem.split("_s", 1)[1].split("_", 1)[0]
    return stem


def write_geotiff(output_path, array, crs, transform, nodata):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="LZW",
    ) as dst:
        dst.write(array, 1)


def find_target_viirs_day_tif(ts_root, fire_id):
    viirs_dir = Path(ts_root) / fire_id / "VIIRS_Day"
    if not viirs_dir.exists():
        raise FileNotFoundError(f"Missing VIIRS_Day directory for {fire_id}: {viirs_dir}")

    tif_files = sorted(viirs_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No VIIRS_Day GeoTIFFs found for {fire_id}: {viirs_dir}")
    return tif_files[0]


def read_target_grid(ts_root, fire_id):
    target_tif = find_target_viirs_day_tif(ts_root, fire_id)
    with rasterio.open(target_tif) as src:
        return {
            "path": str(target_tif),
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
        }


def main():
    args = parse_args()

    rows = load_rows(args.bbox_csvs)
    if args.limit_events > 0:
        rows = rows[: args.limit_events]

    print(f"Using TS-SatFire target grids from: {args.ts_root}")

    summary_rows = []

    for idx, row in enumerate(rows, start=1):
        fire_id = row["Id"]
        year = row["_year"]
        start_date = dt.date.fromisoformat(row["start_date"])
        end_date = dt.date.fromisoformat(row["end_date"])

        try:
            target_grid = read_target_grid(args.ts_root, fire_id)
        except Exception as exc:
            print(f"[ERROR] {fire_id} target grid lookup failed: {exc}")
            summary_rows.append(
                {
                    "year": year,
                    "Id": fire_id,
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "input_files": 0,
                    "saved_files": 0,
                    "existing_files": 0,
                    "empty_files": 0,
                    "error_files": 1,
                    "target_height": "",
                    "target_width": "",
                    "target_tif": "",
                }
            )
            continue

        event_files = list_event_files(args.goes_root, start_date, end_date)
        saved_files = 0
        existing_files = 0
        empty_files = 0
        error_files = 0

        print(
            f"[EVENT {idx}/{len(rows)}] {year} {fire_id} files={len(event_files)} "
            f"target={target_grid['height']}x{target_grid['width']}"
        )

        for nc_path in event_files:
            try:
                ds = xr.open_dataset(nc_path)
                clipped = clip_dataset(ds, compute_xy_bounds(ds, row))
                clip_h = int(clipped.sizes.get("y", 0))
                clip_w = int(clipped.sizes.get("x", 0))
                if clip_h == 0 or clip_w == 0:
                    ds.close()
                    empty_files += 1
                    continue

                source_crs = build_goes_crs(clipped)
                source_transform = build_source_transform(clipped)
                timestamp = extract_timestamp(nc_path, clipped)

                mask_fill = clipped["Mask"].attrs.get("_FillValue", -99)
                mask_src = clipped["Mask"].values
                if np.issubdtype(mask_src.dtype, np.floating):
                    mask_src = np.where(np.isnan(mask_src), mask_fill, mask_src)
                mask_src = mask_src.astype(np.int16, copy=False)
                mask_dst = np.full(
                    (target_grid["height"], target_grid["width"]),
                    np.int16(mask_fill),
                    dtype=np.int16,
                )

                frp_fill = clipped["Power"].attrs.get("_FillValue", -9999.0)
                frp_src = clipped["Power"].values
                frp_src = np.where(np.isnan(frp_src), frp_fill, frp_src).astype(np.float32, copy=False)
                frp_dst = np.full(
                    (target_grid["height"], target_grid["width"]),
                    np.float32(frp_fill),
                    dtype=np.float32,
                )

                reproject(
                    source=mask_src,
                    destination=mask_dst,
                    src_transform=source_transform,
                    src_crs=source_crs,
                    dst_transform=target_grid["transform"],
                    dst_crs=target_grid["crs"],
                    src_nodata=np.int16(mask_fill),
                    dst_nodata=np.int16(mask_fill),
                    resampling=Resampling.nearest,
                )

                reproject(
                    source=frp_src,
                    destination=frp_dst,
                    src_transform=source_transform,
                    src_crs=source_crs,
                    dst_transform=target_grid["transform"],
                    dst_crs=target_grid["crs"],
                    src_nodata=np.float32(frp_fill),
                    dst_nodata=np.float32(frp_fill),
                    resampling=Resampling.nearest,
                )

                mask_path = (
                    Path(args.output_root)
                    / year
                    / fire_id
                    / "mask_fixed"
                    / f"{timestamp}_mask.tif"
                )
                frp_path = (
                    Path(args.output_root)
                    / year
                    / fire_id
                    / "frp_fixed"
                    / f"{timestamp}_frp.tif"
                )

                if mask_path.exists() and frp_path.exists() and not args.overwrite:
                    ds.close()
                    existing_files += 1
                    continue

                write_geotiff(
                    mask_path,
                    mask_dst,
                    target_grid["crs"],
                    target_grid["transform"],
                    np.int16(mask_fill),
                )
                write_geotiff(
                    frp_path,
                    frp_dst,
                    target_grid["crs"],
                    target_grid["transform"],
                    np.float32(frp_fill),
                )
                ds.close()
                saved_files += 1

            except Exception as exc:
                error_files += 1
                print(f"[ERROR] {fire_id} {nc_path}: {exc}")

        summary_rows.append(
            {
                "year": year,
                "Id": fire_id,
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "input_files": len(event_files),
                "saved_files": saved_files,
                "existing_files": existing_files,
                "empty_files": empty_files,
                "error_files": error_files,
                "target_height": target_grid["height"],
                "target_width": target_grid["width"],
                "target_tif": target_grid["path"],
            }
        )

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "Id",
                "start_date",
                "end_date",
                "input_files",
                "saved_files",
                "existing_files",
                "empty_files",
                "error_files",
                "target_height",
                "target_width",
                "target_tif",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
