import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.infer_real_oco3_absolute_by_plant import infer_one_plant


PLANTS = [
    "Bełchatów",
    "Labadie",
    "Parish",
    "Cumberland",
    "Intermountain",
    "Tuoketuo",
    "Colstrip",
    "Conemaugh",
]

ANNUAL_EMISSIONS = {
    "Belchatow": 37.6,
    "Labadie": 15.0,
    "Parish": 13.2,
    "Cumberland": 12.4,
    "Intermountain": 5.0,
    "Tuoketuo": 29.5,
    "Colstrip": 13.6,
    "Conemaugh": 16.9,
}

PLANT_ORDER = [
    "Belchatow",
    "Labadie",
    "Parish",
    "Cumberland",
    "Intermountain",
    "Tuoketuo",
    "Colstrip",
    "Conemaugh",
]


def canonical_plant_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    if name in ("Bełchatów", "Be艂chat贸w"):
        return "Belchatow"
    return name


def box_stats(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    inliers = arr[(arr >= low_fence) & (arr <= high_fence)]
    whisker_low = float(inliers.min()) if inliers.size else float(arr.min())
    whisker_high = float(inliers.max()) if inliers.size else float(arr.max())
    outliers = arr[(arr < whisker_low) | (arr > whisker_high)]
    return {
        "q1": float(q1),
        "med": float(med),
        "q3": float(q3),
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "outliers": outliers.tolist(),
    }


def plot_ensemble_boxplot(ensemble_scene: pd.DataFrame, out_path: Path):
    all_vals = pd.concat(
        [
            ensemble_scene["prediction_mean"].astype(float),
            pd.Series(list(ANNUAL_EMISSIONS.values()), dtype=float),
        ],
        ignore_index=True,
    )
    y_min = float(np.floor((all_vals.min() - 2) / 5.0) * 5.0)
    y_max = float(np.ceil((all_vals.max() + 2) / 5.0) * 5.0)

    width, height = 2200, 960
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 70, 125
    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def map_y(v):
        span = max(y_max - y_min, 1e-6)
        ratio = (float(v) - y_min) / span
        return plot_bottom - ratio * (plot_bottom - plot_top)

    def draw_box(x_center, width_box, stats, color):
        if stats is None:
            return
        q1 = map_y(stats["q1"])
        med = map_y(stats["med"])
        q3 = map_y(stats["q3"])
        wl = map_y(stats["whisker_low"])
        wh = map_y(stats["whisker_high"])
        left = x_center - width_box / 2
        right = x_center + width_box / 2
        draw.rectangle([left, q3, right, q1], outline=color, width=3)
        draw.line([left, med, right, med], fill=color, width=3)
        draw.line([x_center, q3, x_center, wh], fill=color, width=2)
        draw.line([x_center, q1, x_center, wl], fill=color, width=2)
        cap = width_box * 0.35
        draw.line([x_center - cap / 2, wh, x_center + cap / 2, wh], fill=color, width=2)
        draw.line([x_center - cap / 2, wl, x_center + cap / 2, wl], fill=color, width=2)
        for outlier in stats["outliers"]:
            y = map_y(outlier)
            r = 4
            draw.ellipse([x_center - r, y - r, x_center + r, y + r], outline=color, fill=color)

    for tick in np.arange(y_min, y_max + 0.1, 5.0):
        y = map_y(tick)
        draw.line([plot_left, y, plot_right, y], fill=(220, 220, 220), width=1)
        draw.text((20, y - 7), f"{tick:.0f}", fill="black", font=font)

    draw.line([plot_left, plot_top, plot_left, plot_bottom], fill="black", width=2)
    draw.line([plot_left, plot_bottom, plot_right, plot_bottom], fill="black", width=2)

    group_span = (plot_right - plot_left) / len(PLANT_ORDER)
    box_width = 26
    box_color = (166, 217, 106)
    annual_color = (220, 0, 0)

    for idx, plant in enumerate(PLANT_ORDER):
        center = plot_left + group_span * (idx + 0.5)
        vals = ensemble_scene.loc[
            ensemble_scene["plant_name_canonical"] == plant, "prediction_mean"
        ].dropna().tolist()
        draw_box(center, box_width, box_stats(vals), box_color)
        y_ref = map_y(ANNUAL_EMISSIONS[plant])
        draw.line([center - 45, y_ref, center + 45, y_ref], fill=annual_color, width=3)
        label_w = draw.textlength(plant, font=font)
        draw.text((center - label_w / 2, plot_bottom + 20), plant, fill="black", font=font)

    title = "make6 bs64 5-seed ensemble on fixed 8-plant test set"
    title_w = draw.textlength(title, font=font)
    draw.text(((width - title_w) / 2, 20), title, fill="black", font=font)

    legend_y = plot_top + 10
    draw.line([plot_right - 320, legend_y, plot_right - 280, legend_y], fill=box_color, width=3)
    draw.text((plot_right - 270, legend_y - 7), "ensemble scene mean", fill="black", font=font)
    draw.line([plot_right - 110, legend_y, plot_right - 70, legend_y], fill=annual_color, width=3)
    draw.text((plot_right - 60, legend_y - 7), "annual", fill="black", font=font)

    image.save(out_path)


def build_plant_metrics(ensemble_scene: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for plant in PLANT_ORDER:
        dfp = ensemble_scene[ensemble_scene["plant_name_canonical"] == plant].copy()
        annual = ANNUAL_EMISSIONS[plant]
        pred_mean = float(dfp["prediction_mean"].mean())
        pred_median = float(dfp["prediction_mean"].median())
        mae_mean = abs(pred_mean - annual)
        mae_median = abs(pred_median - annual)
        mape_mean = 100.0 * mae_mean / annual
        mape_median = 100.0 * mae_median / annual
        rows.append(
            {
                "plant_name": plant,
                "n_scenes": len(dfp),
                "annual_emission_mt_y": annual,
                "ensemble_scene_mean": pred_mean,
                "ensemble_scene_median": pred_median,
                "mae_mean": mae_mean,
                "mape_mean_pct": mape_mean,
                "mae_median": mae_median,
                "mape_median_pct": mape_median,
                "scene_mean_std_across_seeds_avg": float(dfp["prediction_std"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    overall = {
        "plant_name": "OVERALL",
        "n_scenes": int(out["n_scenes"].sum()),
        "annual_emission_mt_y": np.nan,
        "ensemble_scene_mean": np.nan,
        "ensemble_scene_median": np.nan,
        "mae_mean": float(out["mae_mean"].mean()),
        "mape_mean_pct": float(out["mape_mean_pct"].mean()),
        "mae_median": float(out["mae_median"].mean()),
        "mape_median_pct": float(out["mape_median_pct"].mean()),
        "scene_mean_std_across_seeds_avg": float(out["scene_mean_std_across_seeds_avg"].mean()),
    }
    return pd.concat([out, pd.DataFrame([overall])], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Run 5-seed make6 bs64 fixed-test ensemble report.")
    parser.add_argument("--model-dirs", nargs="+", required=True)
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--nc4-root", required=True)
    parser.add_argument("--weights-name", default="w_best.h5")
    parser.add_argument("--field-name", default="xco2_mixed_absolute")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_h5 = Path(args.train_h5)
    nc4_root = Path(args.nc4_root)

    frames: List[pd.DataFrame] = []
    for model_dir_str in args.model_dirs:
        model_dir = Path(model_dir_str)
        seed_tag = model_dir.name.split("_s")[-1] if "_s" in model_dir.name else model_dir.name
        model_out_dir = out_dir / f"seed_{seed_tag}"
        model_out_dir.mkdir(parents=True, exist_ok=True)
        for plant in PLANTS:
            plant_dir = nc4_root / plant
            df = infer_one_plant(
                model_dir=model_dir,
                train_h5=train_h5,
                plant_dir=plant_dir,
                plant_name=plant,
                field_name=args.field_name,
                weights_name=args.weights_name,
                out_dir=model_out_dir,
                skip_plots=True if args.skip_plots else False,
            )
            df["seed_tag"] = seed_tag
            df["model_dir"] = str(model_dir)
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["plant_name_canonical"] = combined["plant_name"].map(canonical_plant_name)
    combined.to_csv(out_dir / "all_seed_predictions.csv", index=False)

    ensemble_scene = (
        combined.groupby(["plant_name_canonical", "file_name"], as_index=False)
        .agg(
            prediction_mean=("predicted_emission", "mean"),
            prediction_median=("predicted_emission", "median"),
            prediction_std=("predicted_emission", "std"),
            valid_pixels=("valid_pixels", "first"),
            n_seeds=("seed_tag", "count"),
        )
    )
    ensemble_scene["prediction_std"] = ensemble_scene["prediction_std"].fillna(0.0)
    ensemble_scene.to_csv(out_dir / "ensemble_scene_predictions.csv", index=False)

    plant_metrics = build_plant_metrics(ensemble_scene)
    plant_metrics.to_csv(out_dir / "ensemble_mape_mae_table.csv", index=False)

    plot_ensemble_boxplot(
        ensemble_scene,
        out_dir / "ensemble_boxplot_with_annual_lines.png",
    )

    print("Saved:", out_dir / "all_seed_predictions.csv")
    print("Saved:", out_dir / "ensemble_scene_predictions.csv")
    print("Saved:", out_dir / "ensemble_mape_mae_table.csv")
    print("Saved:", out_dir / "ensemble_boxplot_with_annual_lines.png")


if __name__ == "__main__":
    main()
