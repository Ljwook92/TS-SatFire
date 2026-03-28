import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.transform import Affine


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clip GOES FDCF NetCDF files using one common max bbox across multiple "
            "TS-SatFire bbox CSVs, save mask/frp GeoTIFFs with corrected meter-based "
            "geostationary coordinates, and pad smaller outputs to a common raster size."
        )
    )
    parser.add_argument("--goes-root", required=True, help="Root directory of GOES FDCF NetCDF files.")
    parser.add_argument(
        "--bbox-csvs",
        nargs="+",
        required=True,
        help="List of bbox CSV paths, e.g. us_fire_2019_bbox.csv us_fire_2020_bbox.csv us_fire_2021_bbox.csv",
    )
    parser.add_argument("--output-root", required=True, help="Output root directory.")
    parser.add_argument("--summary-csv", required=True, help="Path to write the summary CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing GeoTIFF outputs.")
    parser.add_argument("--limit-events", type=int, default=0, help="Optional limit on total events to process.")
    return parser.parse_args()


def km_to_lat_deg(km):
    return km / 110.574


def km_to_lon_deg(km, lat):
    cos_lat = np.cos(np.radians(lat))
    if abs(cos_lat) < 1e-8:
        raise ValueError(f"Longitude conversion unstable at latitude={lat}")
    return km / (111.320 * cos_lat)


def daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def infer_year_from_csv_path(csv_path):
    name = Path(csv_path).stem
    parts = name.split("_")
    for part in parts:
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


def compute_common_half_sizes(rows):
    max_half_width = max(float(row["half_width_km"]) for row in rows)
    max_half_height = max(float(row["half_height_km"]) for row in rows)
    return max_half_width, max_half_height


def apply_common_bbox(row, common_half_width_km, common_half_height_km):
    center_lat = float(row["center_lat"])
    center_lon = float(row["center_lon"])
    lat_offset = km_to_lat_deg(common_half_height_km)
    lon_offset = km_to_lon_deg(common_half_width_km, center_lat)

    updated = dict(row)
    updated["min_lon"] = center_lon - lon_offset
    updated["max_lon"] = center_lon + lon_offset
    updated["min_lat"] = center_lat - lat_offset
    updated["max_lat"] = center_lat + lat_offset
    updated["common_half_width_km"] = common_half_width_km
    updated["common_half_height_km"] = common_half_height_km
    return updated


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


def build_affine_transform(clipped):
    x = clipped.x.values
    y = clipped.y.values
    if x.size < 2 or y.size < 2:
        raise ValueError("Need at least two x/y coordinates to build the transform.")

    sat_height = clipped["goes_imager_projection"].perspective_point_height.item()
    x_m = x * sat_height
    y_m = y * sat_height

    x_res = float(x_m[1] - x_m[0])
    y_res = float(y_m[1] - y_m[0])
    x_origin = float(x_m[0] - x_res / 2.0)
    y_origin = float(y_m[0] - y_res / 2.0)
    return Affine(x_res, 0.0, x_origin, 0.0, y_res, y_origin)


def list_event_files(goes_root, start_date, end_date):
    files = []
    for current_date in daterange(start_date, end_date):
        year = f"{current_date.year:04d}"
        doy = f"{current_date.timetuple().tm_yday:03d}"
        day_dir = Path(goes_root) / year / doy
        if day_dir.exists():
            files.extend(sorted(day_dir.rglob("*.nc")))
    return files


def first_event_file(goes_root, row):
    files = list_event_files(
        goes_root,
        dt.date.fromisoformat(row["start_date"]),
        dt.date.fromisoformat(row["end_date"]),
    )
    return files[0] if files else None


def extract_timestamp(nc_path, clipped):
    time_str = clipped.attrs.get("time_coverage_start", "")
    if time_str:
        return (
            time_str.replace("-", "")
            .replace(":", "")
            .split(".")[0]
        )
    stem = Path(nc_path).stem
    if "_s" in stem:
        return stem.split("_s", 1)[1].split("_", 1)[0]
    return stem


def pad_array(array, target_height, target_width, fill_value):
    height, width = array.shape
    output = np.full((target_height, target_width), fill_value, dtype=array.dtype)
    y0 = (target_height - height) // 2
    x0 = (target_width - width) // 2
    output[y0:y0 + height, x0:x0 + width] = array
    return output


def write_geotiff(output_path, array, crs_wkt, transform, nodata):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs_wkt,
        transform=transform,
        nodata=nodata,
        compress="LZW",
    ) as dst:
        dst.write(array, 1)


def determine_target_shape(goes_root, rows):
    max_height = 0
    max_width = 0

    for row in rows:
        sample_nc = first_event_file(goes_root, row)
        if sample_nc is None:
            continue

        ds = xr.open_dataset(sample_nc)
        clipped = clip_dataset(ds, compute_xy_bounds(ds, row))
        height = int(clipped.sizes.get("y", 0))
        width = int(clipped.sizes.get("x", 0))
        ds.close()

        max_height = max(max_height, height)
        max_width = max(max_width, width)

    if max_height == 0 or max_width == 0:
        raise RuntimeError("Could not determine a valid target raster shape.")

    return max_height, max_width


def main():
    args = parse_args()

    rows = load_rows(args.bbox_csvs)
    if args.limit_events > 0:
        rows = rows[: args.limit_events]

    common_half_width_km, common_half_height_km = compute_common_half_sizes(rows)
    common_rows = [
        apply_common_bbox(row, common_half_width_km, common_half_height_km)
        for row in rows
    ]

    print(
        f"Common half sizes (km): width={common_half_width_km}, "
        f"height={common_half_height_km}"
    )

    target_height, target_width = determine_target_shape(args.goes_root, common_rows)
    print(f"Target padded raster size: {target_height} x {target_width}")

    summary_rows = []

    for idx, row in enumerate(common_rows, start=1):
        fire_id = row["Id"]
        year = row["_year"]
        start_date = dt.date.fromisoformat(row["start_date"])
        end_date = dt.date.fromisoformat(row["end_date"])
        event_files = list_event_files(args.goes_root, start_date, end_date)

        saved_files = 0
        existing_files = 0
        empty_files = 0
        error_files = 0

        print(f"[EVENT {idx}/{len(common_rows)}] {year} {fire_id} files={len(event_files)}")

        for nc_path in event_files:
            try:
                ds = xr.open_dataset(nc_path)
                clipped = clip_dataset(ds, compute_xy_bounds(ds, row))

                height = int(clipped.sizes.get("y", 0))
                width = int(clipped.sizes.get("x", 0))
                if height == 0 or width == 0:
                    ds.close()
                    empty_files += 1
                    continue

                crs_wkt = build_goes_crs(clipped).to_wkt()
                transform = build_affine_transform(clipped)
                timestamp = extract_timestamp(nc_path, clipped)

                mask_fill = clipped["Mask"].attrs.get("_FillValue", -99)
                mask = clipped["Mask"].values
                if np.issubdtype(mask.dtype, np.floating):
                    mask = np.where(np.isnan(mask), mask_fill, mask)
                mask = mask.astype(np.int16, copy=False)

                frp_fill = clipped["Power"].attrs.get("_FillValue", -9999.0)
                frp = clipped["Power"].values
                frp = np.where(np.isnan(frp), frp_fill, frp).astype(np.float32, copy=False)

                mask = pad_array(mask, target_height, target_width, np.int16(mask_fill))
                frp = pad_array(frp, target_height, target_width, np.float32(frp_fill))

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

                write_geotiff(mask_path, mask, crs_wkt, transform, mask_fill)
                write_geotiff(frp_path, frp, crs_wkt, transform, float(frp_fill))
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
                "target_height": target_height,
                "target_width": target_width,
                "common_half_width_km": common_half_width_km,
                "common_half_height_km": common_half_height_km,
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
                "common_half_width_km",
                "common_half_height_km",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
