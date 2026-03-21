import argparse
import csv
import datetime as dt
from pathlib import Path

import xarray as xr
from pyproj import CRS, Transformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clip all GOES-R FDCF files matching bbox row date ranges."
    )
    parser.add_argument("--goes-root", required=True, help="Root directory of downloaded GOES product.")
    parser.add_argument("--bbox-csv", required=True, help="Path to bbox CSV.")
    parser.add_argument("--output-root", required=True, help="Directory to store clipped outputs.")
    parser.add_argument(
        "--limit-events",
        type=int,
        default=0,
        help="Optional max number of events to process. 0 means all.",
    )
    parser.add_argument(
        "--limit-files-per-event",
        type=int,
        default=0,
        help="Optional max number of files per event. 0 means all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clipped files.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional path to save per-event clip summary.",
    )
    return parser.parse_args()


def daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


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

    corner_lons = [min_lon, min_lon, max_lon, max_lon]
    corner_lats = [min_lat, max_lat, min_lat, max_lat]
    x_m, y_m = transformer.transform(corner_lons, corner_lats)

    x_rad = [value / sat_height for value in x_m]
    y_rad = [value / sat_height for value in y_m]
    return min(x_rad), max(x_rad), min(y_rad), max(y_rad)


def clip_dataset(ds, xy_bounds):
    x_min, x_max, y_min, y_max = xy_bounds
    x_slice = slice(x_min, x_max)
    if ds.y.values[0] > ds.y.values[-1]:
        y_slice = slice(y_max, y_min)
    else:
        y_slice = slice(y_min, y_max)
    return ds.sel(x=x_slice, y=y_slice)


def clear_netcdf_encoding(ds):
    ds = ds.copy()
    ds.encoding = {}
    for name in ds.coords:
        ds[name].encoding = {}
    for name in ds.data_vars:
        ds[name].encoding = {}
    return ds


def list_event_files(goes_root, start_date, end_date):
    files = []
    for current_date in daterange(start_date, end_date):
        year = f"{current_date.year:04d}"
        doy = f"{current_date.timetuple().tm_yday:03d}"
        day_dir = goes_root / year / doy
        if day_dir.exists():
            files.extend(sorted(day_dir.rglob("*.nc")))
    return files


def clip_one_file(nc_path, row, output_root, overwrite):
    fire_id = row["Id"]
    ds = xr.open_dataset(nc_path)
    xy_bounds = compute_xy_bounds(ds, row)
    clipped = clip_dataset(ds, xy_bounds)

    if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
        ds.close()
        return "empty"

    clipped = clear_netcdf_encoding(clipped)

    out_dir = output_root / fire_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / nc_path.name

    if out_path.exists() and not overwrite:
        ds.close()
        return "exists"

    clipped.to_netcdf(out_path)
    ds.close()
    return "saved"


def main():
    args = parse_args()
    goes_root = Path(args.goes_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(args.bbox_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if args.limit_events > 0:
        rows = rows[: args.limit_events]

    summary_rows = []

    for idx, row in enumerate(rows, start=1):
        fire_id = row["Id"]
        start_date = dt.date.fromisoformat(row["start_date"])
        end_date = dt.date.fromisoformat(row["end_date"])
        event_files = list_event_files(goes_root, start_date, end_date)

        if args.limit_files_per_event > 0:
            event_files = event_files[: args.limit_files_per_event]

        saved_count = 0
        empty_count = 0
        exists_count = 0
        error_count = 0

        print(f"[EVENT {idx}/{len(rows)}] {fire_id} files={len(event_files)}")

        for nc_path in event_files:
            try:
                result = clip_one_file(nc_path, row, output_root, args.overwrite)
            except Exception as exc:
                error_count += 1
                print(f"[ERROR] {fire_id} {nc_path}: {exc}")
                continue

            if result == "saved":
                saved_count += 1
            elif result == "empty":
                empty_count += 1
            elif result == "exists":
                exists_count += 1

        summary_rows.append(
            {
                "Id": fire_id,
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "input_files": len(event_files),
                "saved_files": saved_count,
                "empty_files": empty_count,
                "existing_files": exists_count,
                "error_files": error_count,
            }
        )

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Id",
                    "start_date",
                    "end_date",
                    "input_files",
                    "saved_files",
                    "empty_files",
                    "existing_files",
                    "error_files",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved clip summary to {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
