import argparse
from pathlib import Path

import rasterio
from rasterio.transform import Affine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix GOES GeoTIFF georeferencing by converting fixed-grid radian coordinates to meters."
    )
    parser.add_argument("--input", required=True, help="Input GOES GeoTIFF with radian-based transform.")
    parser.add_argument("--output", required=True, help="Output GeoTIFF path.")
    parser.add_argument(
        "--sat-height",
        type=float,
        default=35786023.0,
        help="GOES perspective point height in meters. Default matches GOES-R metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        data = src.read()

        old_t = src.transform
        new_t = Affine(
            old_t.a * args.sat_height,
            old_t.b,
            old_t.c * args.sat_height,
            old_t.d,
            old_t.e * args.sat_height,
            old_t.f * args.sat_height,
        )

        profile.update(transform=new_t)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
            dst.update_tags(**src.tags())
            for idx in src.indexes:
                dst.update_tags(idx, **src.tags(idx))

    print(f"Saved fixed GeoTIFF to {output_path}")


if __name__ == "__main__":
    main()
