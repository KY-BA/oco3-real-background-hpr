import argparse
from pathlib import Path

import h5py
import numpy as np
import xarray as xr


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_dataset(src_handle, dst_handle, key):
    src_handle.copy(key, dst_handle)


def add_uv_field(
    src_h5: Path,
    dst_h5: Path,
    plume_dataset: Path,
    src_field: str,
    dst_field: str,
    chunk_size: int,
) -> None:
    dst_h5.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(plume_dataset)
    u_all = np.asarray(ds["u"].values, dtype=np.float32)
    v_all = np.asarray(ds["v"].values, dtype=np.float32)

    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        copy_attrs(src, dst)
        dst.attrs["field_name"] = dst_field
        dst.attrs["uv_source_dataset"] = plume_dataset.as_posix()
        dst.attrs["uv_source_variables"] = "u,v"

        for key in src.keys():
            if key == src_field:
                continue
            copy_dataset(src, dst, key)

        x_src = src[src_field]
        source_idx = src["source_idx_img"]
        if x_src.ndim != 3:
            raise ValueError(f"{src_field} must be 3D, got shape {x_src.shape}")

        total, height, width = x_src.shape
        x_dst = dst.create_dataset(
            dst_field,
            shape=(total, height, width, 3),
            dtype=np.float32,
            chunks=(1, height, width, 3),
            compression="gzip",
        )

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            idx = np.asarray(source_idx[start:end], dtype=np.int64)
            stacked = np.empty((end - start, height, width, 3), dtype=np.float32)
            stacked[..., 0] = np.asarray(x_src[start:end], dtype=np.float32)
            stacked[..., 1] = u_all[idx]
            stacked[..., 2] = v_all[idx]
            x_dst[start:end] = stacked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create xco2/u/v mixed H5 files from an existing absolute mixed dataset."
    )
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--plume-dataset", required=True)
    parser.add_argument("--src-field", default="xco2_mixed_absolute")
    parser.add_argument("--dst-field", default="xco2_mixed_absolute_uv")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "valid_2", "extra_valid"])
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        src_h5 = src_dir / f"{split}.h5"
        dst_h5 = out_dir / f"{split}.h5"
        print(f"Writing {dst_h5}")
        add_uv_field(
            src_h5=src_h5,
            dst_h5=dst_h5,
            plume_dataset=Path(args.plume_dataset),
            src_field=args.src_field,
            dst_field=args.dst_field,
            chunk_size=args.chunk_size,
        )

    for name in ["background_manifest.csv", "README.txt"]:
        src_path = src_dir / name
        if src_path.exists():
            dst_path = out_dir / name
            text = src_path.read_text(encoding="utf-8")
            if name == "README.txt":
                text = text.replace(
                    f"field_name={args.src_field}",
                    f"field_name={args.dst_field}",
                )
                text += "\nuv_extension=xco2 channel from mixed absolute; u/v channels from source plume dataset by source_idx_img\n"
            dst_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
