import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_ch4_1_cnn_background_transfer import (
    build_overall_metrics,
    build_plant_metrics,
    build_scene_predictions,
    evaluate_mixed_variant,
)


def emission_group(annual: float) -> str:
    if annual <= 8:
        return "Low"
    if annual <= 18:
        return "Medium"
    return "High"


def group_diagnostics(plant_df: pd.DataFrame) -> pd.DataFrame:
    df = plant_df.copy()
    df["emission_group"] = df["annual_emission_mt_y"].map(emission_group)
    rows = []
    for variant in df["variant"].drop_duplicates():
        dfv = df[df["variant"] == variant]
        for group in ["Low", "Medium", "High"]:
            dfg = dfv[dfv["emission_group"] == group]
            if dfg.empty:
                continue
            rows.append(
                {
                    "variant": variant,
                    "emission_group": group,
                    "n_plants": int(len(dfg)),
                    "n_scenes": int(dfg["n_scenes"].sum()),
                    "mean_bias": float(dfg["bias_mean"].mean()),
                    "mae": float(dfg["ae_mean"].mean()),
                    "mape_pct": float(dfg["ape_mean_pct"].mean()),
                    "median_ape_pct": float(dfg["ape_mean_pct"].median()),
                    "underestimation_rate_pct": float((dfg["bias_mean"] < 0).mean() * 100.0),
                    "plants": ", ".join(dfg["plant_name"].tolist()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final Chapter 4.3 HPR 5-seed inference and CNN comparison."
    )
    parser.add_argument("--hpr-model-dirs", nargs="+", required=True)
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--nc4-root", required=True)
    parser.add_argument("--field-name", default="xco2_mixed_absolute_uv")
    parser.add_argument("--weights-name", default="w_best.h5")
    parser.add_argument("--cnn-plant-metrics", required=True)
    parser.add_argument("--cnn-overall-metrics", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hpr_seed = evaluate_mixed_variant(
        args.hpr_model_dirs,
        Path(args.train_h5),
        Path(args.nc4_root),
        args.field_name,
        args.weights_name,
        "HPR",
    )
    hpr_scene = build_scene_predictions(hpr_seed)
    hpr_plant = build_plant_metrics(hpr_scene)
    hpr_overall = build_overall_metrics(hpr_plant)

    cnn_plant = pd.read_csv(args.cnn_plant_metrics)
    cnn_plant = cnn_plant[cnn_plant["variant"] == "Real-background mixed CNN"].copy()
    cnn_overall = pd.read_csv(args.cnn_overall_metrics)
    cnn_overall = cnn_overall[cnn_overall["variant"] == "Real-background mixed CNN"].copy()

    combined_plant = pd.concat([cnn_plant, hpr_plant], ignore_index=True)
    combined_overall = pd.concat([cnn_overall, hpr_overall], ignore_index=True)
    groups = group_diagnostics(combined_plant)

    hpr_seed.to_csv(out_dir / "hpr_seed_level_predictions.csv", index=False)
    hpr_scene.to_csv(out_dir / "hpr_scene_level_ensemble_predictions.csv", index=False)
    hpr_plant.to_csv(out_dir / "hpr_plant_level_metrics.csv", index=False)
    hpr_overall.to_csv(out_dir / "hpr_overall_metrics.csv", index=False)
    combined_plant.to_csv(out_dir / "cnn_hpr_plant_level_metrics.csv", index=False)
    combined_overall.to_csv(out_dir / "cnn_hpr_overall_metrics.csv", index=False)
    groups.to_csv(out_dir / "cnn_hpr_group_diagnostics.csv", index=False)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "hpr_model_dirs": [str(Path(p)) for p in args.hpr_model_dirs],
                "train_h5": str(Path(args.train_h5)),
                "nc4_root": str(Path(args.nc4_root)),
                "field_name": args.field_name,
                "weights_name": args.weights_name,
                "cnn_plant_metrics": str(Path(args.cnn_plant_metrics)),
                "cnn_overall_metrics": str(Path(args.cnn_overall_metrics)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved outputs to", out_dir)
    print(combined_overall.to_string(index=False))
    print(groups.to_string(index=False))


if __name__ == "__main__":
    main()
