import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_puresim_vs_mixed_on_real_oco3 import (
    apply_norm,
    fill_per_channel_median,
    grid_three_channels,
    load_cfg_fields,
    load_model as load_puresim_model,
    load_norm_spec,
    load_raw_points,
)
from tools.infer_real_oco3_absolute import (
    compute_fill_values,
    load_model as load_mixed_model,
    load_norm_model,
    prepare_input_from_nc4,
    select_emission_output,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANNUAL_EMISSIONS = {
    "Belchatow": 37.6,
    "Tuoketuo": 29.5,
    "Conemaugh": 16.9,
    "Labadie": 15.0,
    "Colstrip": 13.6,
    "Parish": 13.2,
    "Cumberland": 12.4,
    "Intermountain": 5.0,
}

PLANT_COORDS = {
    "Belchatow": (51.268, 19.331),
    "Tuoketuo": (40.203, 111.237),
    "Conemaugh": (40.388, -79.342),
    "Labadie": (38.673, -90.851),
    "Colstrip": (45.884, -106.630),
    "Parish": (29.493, -95.627),
    "Cumberland": (36.389, -87.650),
    "Intermountain": (39.364, -112.535),
}

PLANT_ORDER = [
    "Belchatow",
    "Tuoketuo",
    "Conemaugh",
    "Labadie",
    "Colstrip",
    "Parish",
    "Cumberland",
    "Intermountain",
]


def canonical_plant_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    if name == "Belchatow":
        return "Belchatow"
    if name.startswith("Be") and "chat" in name:
        return "Belchatow"
    return name


def find_plant_dirs(nc4_root: Path) -> Dict[str, Path]:
    dirs = {}
    for path in nc4_root.iterdir():
        if not path.is_dir():
            continue
        key = canonical_plant_name(path.name)
        if key in ANNUAL_EMISSIONS:
            dirs[key] = path
    missing = [plant for plant in PLANT_ORDER if plant not in dirs]
    if missing:
        raise FileNotFoundError(f"Missing plant directories for: {missing}")
    return dirs


def read_file_list(plant_dir: Path) -> List[str]:
    if plant_dir.name == "Intermountain":
        filtered = plant_dir / "file_list_valid200.txt"
        if filtered.exists():
            return [
                line.strip().lstrip("\ufeff")
                for line in filtered.read_text(encoding="utf-8").splitlines()
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


def seed_from_model_dir(model_dir: Path) -> str:
    name = model_dir.name
    if "_s" in name:
        return name.rsplit("_s", 1)[-1]
    return name


def build_puresim_input(cfg_fields, xco2_grid, u_grid, v_grid):
    channel_map = {
        "xco2": xco2_grid,
        "u_wind": u_grid,
        "u_wind_train": u_grid,
        "v_wind": v_grid,
        "v_wind_train": v_grid,
    }
    channels = []
    for key in ["chan_0", "chan_1", "chan_2", "chan_3", "chan_4"]:
        chan = str(cfg_fields.get(key, "None"))
        if chan == "None":
            continue
        if chan not in channel_map:
            raise ValueError(f"Unsupported pure-sim real-test channel: {chan}")
        channels.append(channel_map[chan])
    if not channels:
        raise ValueError("No input channels found in pure-sim model config.")
    return fill_per_channel_median(np.stack(channels, axis=-1))


def evaluate_puresim_variant(model_dirs, nc4_root: Path, weights_name: str, label: str):
    plant_dirs = find_plant_dirs(nc4_root)
    rows = []
    for model_dir in model_dirs:
        model_dir = Path(model_dir)
        cfg_fields = load_cfg_fields(model_dir)
        norm_spec = load_norm_spec(model_dir)
        model = load_puresim_model(model_dir, weights_name)
        seed = seed_from_model_dir(model_dir)
        for plant in PLANT_ORDER:
            plant_lat, plant_lon = PLANT_COORDS[plant]
            for filename in read_file_list(plant_dirs[plant]):
                nc4_path = plant_dirs[plant] / filename
                points = load_raw_points(nc4_path, quality_filter=True)
                xco2_grid, u_grid, v_grid, valid_mask = grid_three_channels(
                    points, lat0=plant_lat, lon0=plant_lon
                )
                x_raw = build_puresim_input(cfg_fields, xco2_grid, u_grid, v_grid)
                x_norm = apply_norm(x_raw[np.newaxis, ...], norm_spec)
                pred = float(np.squeeze(model.predict(x_norm, verbose=0)))
                rows.append(
                    {
                        "variant": label,
                        "seed": seed,
                        "model_dir": str(model_dir),
                        "plant_name": plant,
                        "file_name": filename,
                        "predicted_emission": pred,
                        "valid_pixels": int(valid_mask.sum()),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_mixed_variant(
    model_dirs,
    train_h5: Path,
    nc4_root: Path,
    field_name: str,
    weights_name: str,
    label: str,
):
    plant_dirs = find_plant_dirs(nc4_root)
    fill_values = compute_fill_values(train_h5, field_name)
    rows = []
    for model_dir in model_dirs:
        model_dir = Path(model_dir)
        norm_model = load_norm_model(model_dir)
        model = load_mixed_model(model_dir, weights_name)
        seed = seed_from_model_dir(model_dir)
        for plant in PLANT_ORDER:
            plant_lat, plant_lon = PLANT_COORDS[plant]
            for filename in read_file_list(plant_dirs[plant]):
                prepared = prepare_input_from_nc4(
                    plant_dirs[plant] / filename,
                    field_name=field_name,
                    fill_values=fill_values,
                    norm_model=norm_model,
                    plant_lat=plant_lat,
                    plant_lon=plant_lon,
                )
                pred = float(
                    np.squeeze(
                        select_emission_output(
                            model.predict(prepared["x_norm"], verbose=0)
                        )
                    )
                )
                rows.append(
                    {
                        "variant": label,
                        "seed": seed,
                        "model_dir": str(model_dir),
                        "plant_name": plant,
                        "file_name": filename,
                        "predicted_emission": pred,
                        "valid_pixels": int(prepared["valid_mask"].sum()),
                    }
                )
    return pd.DataFrame(rows)


def build_scene_predictions(seed_df: pd.DataFrame) -> pd.DataFrame:
    scene = (
        seed_df.groupby(["variant", "plant_name", "file_name"], as_index=False)
        .agg(
            prediction_mean=("predicted_emission", "mean"),
            prediction_median=("predicted_emission", "median"),
            prediction_std=("predicted_emission", "std"),
            valid_pixels=("valid_pixels", "first"),
            n_seeds=("seed", "nunique"),
        )
    )
    scene["prediction_std"] = scene["prediction_std"].fillna(0.0)
    return scene


def build_plant_metrics(scene_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in scene_df["variant"].drop_duplicates():
        for plant in PLANT_ORDER:
            dfp = scene_df[
                (scene_df["variant"] == variant) & (scene_df["plant_name"] == plant)
            ]
            annual = ANNUAL_EMISSIONS[plant]
            pred_mean = float(dfp["prediction_mean"].mean())
            pred_median = float(dfp["prediction_mean"].median())
            ae = abs(pred_mean - annual)
            ape = 100.0 * ae / annual
            ae_median = abs(pred_median - annual)
            ape_median = 100.0 * ae_median / annual
            rows.append(
                {
                    "variant": variant,
                    "plant_name": plant,
                    "n_scenes": int(len(dfp)),
                    "annual_emission_mt_y": annual,
                    "plant_prediction_mean": pred_mean,
                    "plant_prediction_median": pred_median,
                    "bias_mean": pred_mean - annual,
                    "ae_mean": ae,
                    "ape_mean_pct": ape,
                    "ae_median": ae_median,
                    "ape_median_pct": ape_median,
                    "scene_prediction_std_avg": float(dfp["prediction_std"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_overall_metrics(plant_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, dfv in plant_df.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                "MAE": float(dfv["ae_mean"].mean()),
                "MAPE_pct": float(dfv["ape_mean_pct"].mean()),
                "Median_AE": float(dfv["ae_mean"].median()),
                "Median_APE_pct": float(dfv["ape_mean_pct"].median()),
                "Mean_bias": float(dfv["bias_mean"].mean()),
                "N_plants": int(len(dfv)),
                "N_scenes": int(dfv["n_scenes"].sum()),
            }
        )
    return pd.DataFrame(rows)


def save_latex_table(overall_df: pd.DataFrame, out_path: Path):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of training background domain on the real OCO-3 power-plant transfer performance of the essential CNN. Both models use the same architecture, input channels, five random seeds, plant-level aggregation, and fixed real test set; they differ only in whether the training background is synthetic or curated from real OCO-3 scenes. Median values are computed across plants from errors based on plant-level mean predictions.}",
        r"\label{tab:cnn_background_transfer}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Training configuration & MAE & MAPE & Median AE & Median APE \\",
        r" & ($\mathrm{Mt~CO}_2~\mathrm{yr}^{-1}$) & (\%) & ($\mathrm{Mt~CO}_2~\mathrm{yr}^{-1}$) & (\%) \\",
        r"\midrule",
    ]
    label_map = {
        "Pure-simulation CNN": "Pure-simulation CNN",
        "Real-background mixed CNN": "Real-background mixed CNN",
    }
    for _, row in overall_df.iterrows():
        label = label_map.get(row["variant"], row["variant"])
        lines.append(
            f"{label} & {row['MAE']:.2f} & {row['MAPE_pct']:.1f} & "
            f"{row['Median_AE']:.2f} & {row['Median_APE_pct']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_predicted_vs_reported(plant_df: pd.DataFrame, out_pdf: Path, out_png: Path):
    colors = {
        "Pure-simulation CNN": "#7a7a7a",
        "Real-background mixed CNN": "#1f77b4",
    }
    markers = {
        "Pure-simulation CNN": "o",
        "Real-background mixed CNN": "s",
    }
    offsets = {
        "Pure-simulation CNN": -0.18,
        "Real-background mixed CNN": 0.18,
    }

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    max_lim = max(
        float(plant_df["annual_emission_mt_y"].max()),
        float(plant_df["plant_prediction_mean"].max()),
    )
    max_lim = np.ceil((max_lim + 3.0) / 5.0) * 5.0
    ax.plot([0, max_lim], [0, max_lim], color="black", linestyle="--", linewidth=1.2, label="1:1")

    for variant, dfv in plant_df.groupby("variant", sort=False):
        ax.scatter(
            dfv["annual_emission_mt_y"],
            dfv["plant_prediction_mean"],
            s=56,
            marker=markers.get(variant, "o"),
            color=colors.get(variant, "tab:blue"),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            label=variant,
        )

    abbrev = {
        "Belchatow": "BEL",
        "Tuoketuo": "TUO",
        "Conemaugh": "CON",
        "Labadie": "LAB",
        "Colstrip": "COL",
        "Parish": "PAR",
        "Cumberland": "CUM",
        "Intermountain": "INT",
    }
    for _, row in plant_df.iterrows():
        dx = offsets.get(row["variant"], 0.0)
        ax.text(
            row["annual_emission_mt_y"] + dx,
            row["plant_prediction_mean"] + 0.35,
            abbrev[row["plant_name"]],
            fontsize=7.5,
            color=colors.get(row["variant"], "black"),
            ha="center",
        )

    ax.set_xlim(0, max_lim)
    ax.set_ylim(0, max_lim)
    ax.set_xlabel(r"Reported annual emission ($\mathrm{Mt~CO}_2~\mathrm{yr}^{-1}$)")
    ax.set_ylabel(r"Predicted plant-level emission ($\mathrm{Mt~CO}_2~\mathrm{yr}^{-1}$)")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.13, top=0.96)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pure-sim vs real-background mixed essential CNNs for Chapter 4.1."
    )
    parser.add_argument("--puresim-model-dirs", nargs="+", required=True)
    parser.add_argument("--mixed-model-dirs", nargs="+", required=True)
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--nc4-root", required=True)
    parser.add_argument("--field-name", default="xco2_mixed_absolute_uv")
    parser.add_argument("--weights-name", default="w_best.h5")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--make-plot",
        action="store_true",
        help="Also make the scatter plot. Prefer tools/plot_ch4_1_cnn_background_transfer.py if the TensorFlow environment has Matplotlib instability.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    puresim = evaluate_puresim_variant(
        args.puresim_model_dirs,
        Path(args.nc4_root),
        args.weights_name,
        "Pure-simulation CNN",
    )
    mixed = evaluate_mixed_variant(
        args.mixed_model_dirs,
        Path(args.train_h5),
        Path(args.nc4_root),
        args.field_name,
        args.weights_name,
        "Real-background mixed CNN",
    )

    seed_df = pd.concat([puresim, mixed], ignore_index=True)
    scene_df = build_scene_predictions(seed_df)
    plant_df = build_plant_metrics(scene_df)
    overall_df = build_overall_metrics(plant_df)

    seed_df.to_csv(out_dir / "seed_level_predictions.csv", index=False)
    scene_df.to_csv(out_dir / "scene_level_ensemble_predictions.csv", index=False)
    plant_df.to_csv(out_dir / "plant_level_metrics.csv", index=False)
    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False)
    save_latex_table(overall_df, out_dir / "table_cnn_background_transfer.tex")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "puresim_model_dirs": [str(Path(p)) for p in args.puresim_model_dirs],
                "mixed_model_dirs": [str(Path(p)) for p in args.mixed_model_dirs],
                "train_h5": str(Path(args.train_h5)),
                "nc4_root": str(Path(args.nc4_root)),
                "field_name": args.field_name,
                "weights_name": args.weights_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.make_plot:
        plot_predicted_vs_reported(
            plant_df,
            out_dir / "fig_cnn_background_transfer_scatter.pdf",
            out_dir / "fig_cnn_background_transfer_scatter.png",
        )
    print("Saved outputs to", out_dir)
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
