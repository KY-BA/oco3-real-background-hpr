import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.infer_real_oco3_absolute import (
    compute_fill_values,
    load_model,
    load_norm_model,
    make_summary_plot,
    prepare_input_from_nc4,
    select_emission_output,
)


PLANT_COORDS = {
    "Belchatow": (51.268, 19.331),
    "Bełchatów": (51.268, 19.331),
    "Labadie": (38.673, -90.851),
    "Parish": (29.493, -95.627),
    "Cumberland": (36.389, -87.650),
    "Intermountain": (39.364, -112.535),
    "Tuoketuo": (40.203, 111.237),
    "Colstrip": (45.884, -106.630),
    "Conemaugh": (40.388, -79.342),
}


def canonical_plant_name(name: str) -> str:
    if name == "Bełchatów":
        return "Belchatow"
    return name


def read_file_list(plant_dir: Path) -> List[str]:
    if plant_dir.name == "Intermountain":
        filtered_list = plant_dir / "file_list_valid200.txt"
        if filtered_list.exists():
            return [
                line.strip().lstrip("\ufeff")
                for line in filtered_list.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    file_list = plant_dir / "file_list.txt"
    if file_list.exists():
        return [
            line.strip().lstrip("\ufeff")
            for line in file_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return sorted(path.name for path in plant_dir.glob("*.nc4"))


def infer_one_plant(
    model_dir: Path,
    train_h5: Path,
    plant_dir: Path,
    plant_name: str,
    field_name: str,
    weights_name: str,
    out_dir: Path,
    skip_plots: bool,
):
    plant_key = canonical_plant_name(plant_name)
    if plant_key not in PLANT_COORDS:
        raise ValueError(f"Unknown plant {plant_name}")

    plant_lat, plant_lon = PLANT_COORDS[plant_key]
    out_dir.mkdir(parents=True, exist_ok=True)

    fill_values = compute_fill_values(train_h5, field_name)
    norm_model = load_norm_model(model_dir)
    model = load_model(model_dir, weights_name)

    rows = []
    filenames = read_file_list(plant_dir)
    for filename in filenames:
        prepared = prepare_input_from_nc4(
            plant_dir / filename,
            field_name=field_name,
            fill_values=fill_values,
            norm_model=norm_model,
            plant_lat=plant_lat,
            plant_lon=plant_lon,
        )
        pred = float(select_emission_output(model.predict(prepared["x_norm"], verbose=0)).squeeze())
        rows.append(
            {
                "name": filename.replace(".nc4", ""),
                "file_name": filename,
                "prediction": pred,
                "predicted_emission": pred,
                "valid_pixels": int(prepared["valid_mask"].sum()),
                "missing_pixels": int((~prepared["valid_mask"]).sum()),
                "xco2_min": float(prepared["x_raw"][:, :, 0].min()),
                "xco2_max": float(prepared["x_raw"][:, :, 0].max()),
                "plant_name": plant_name,
                "plant_lat": plant_lat,
                "plant_lon": plant_lon,
                **prepared,
            }
        )

    summary_df = pd.DataFrame(
        [
            {
                "plant_name": row["plant_name"],
                "file_name": row["file_name"],
                "name": row["name"],
                "prediction": row["prediction"],
                "predicted_emission": row["predicted_emission"],
                "valid_pixels": row["valid_pixels"],
                "missing_pixels": row["missing_pixels"],
                "xco2_min": row["xco2_min"],
                "xco2_max": row["xco2_max"],
                "plant_lat": row["plant_lat"],
                "plant_lon": row["plant_lon"],
            }
            for row in rows
        ]
    )
    summary_df.to_csv(out_dir / f"{plant_name}_predictions.csv", index=False)
    if not skip_plots:
        make_summary_plot(rows, out_dir / f"{plant_name}_predictions.png", field_name)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Run mixed-model inference by plant folder.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--nc4-root", required=True, help="Root dir containing plant subfolders.")
    parser.add_argument("--plants", nargs="+", required=True)
    parser.add_argument("--weights-name", default="w_best.h5")
    parser.add_argument("--field-name", default="xco2_mixed_absolute")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    train_h5 = Path(args.train_h5)
    nc4_root = Path(args.nc4_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []
    for plant in args.plants:
        plant_dir = nc4_root / plant
        if not plant_dir.exists():
            raise FileNotFoundError(f"Missing plant folder: {plant_dir}")
        df = infer_one_plant(
            model_dir=model_dir,
            train_h5=train_h5,
            plant_dir=plant_dir,
            plant_name=plant,
            field_name=args.field_name,
            weights_name=args.weights_name,
            out_dir=out_dir,
            skip_plots=args.skip_plots,
        )
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(out_dir / "all_plants_predictions.csv", index=False)
    print("Saved:", out_dir / "all_plants_predictions.csv")
    print(
        "Rows:",
        len(combined),
        "Plants:",
        ", ".join(sorted({canonical_plant_name(str(x)) for x in combined["plant_name"].tolist()})),
    )


if __name__ == "__main__":
    main()
