import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile
import zipfile

import h5py
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.preprocessing import TrainingTimeNormalization

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def km_per_deg_lon(lat_deg):
    return 111.32 * np.cos(np.deg2rad(lat_deg))


def to_local_km(latitude, longitude, lat0, lon0):
    x_km = (longitude - lon0) * km_per_deg_lon(lat0)
    y_km = (latitude - lat0) * 110.57
    return x_km, y_km


def make_grid_centers(size, resolution_km):
    half_span = size * resolution_km / 2.0
    return np.linspace(
        -half_span + resolution_km / 2.0,
        half_span - resolution_km / 2.0,
        size,
    )


def grid_weighted_idw(
    x_obs_km, y_obs_km, values, grid_x_km, grid_y_km, radius_km, power=2.0
):
    grid = np.full((grid_y_km.size, grid_x_km.size), np.nan, dtype=np.float32)
    for iy, y0 in enumerate(grid_y_km):
        dy = y_obs_km - y0
        for ix, x0 in enumerate(grid_x_km):
            dx = x_obs_km - x0
            dist = np.sqrt(dx * dx + dy * dy)
            neighborhood = dist < radius_km
            if not np.any(neighborhood):
                continue
            local_dist = dist[neighborhood]
            local_values = values[neighborhood]
            if np.any(local_dist < 1e-6):
                grid[iy, ix] = np.float32(local_values[np.argmin(local_dist)])
                continue
            weights = 1.0 / np.power(local_dist, power)
            grid[iy, ix] = np.float32(np.sum(weights * local_values) / np.sum(weights))
    return grid


def load_raw_points(nc4_path, quality_filter=True):
    with h5py.File(nc4_path, "r") as handle:
        latitude = np.array(handle["latitude"][:]).astype(np.float32).ravel()
        longitude = np.array(handle["longitude"][:]).astype(np.float32).ravel()
        xco2 = np.array(handle["xco2"][:]).astype(np.float32).ravel()
        u_met = np.array(handle["Meteorology"]["windspeed_u_met"][:]).astype(np.float32).ravel()
        v_met = np.array(handle["Meteorology"]["windspeed_v_met"][:]).astype(np.float32).ravel()
        time_vals = np.array(handle["time"][:]).astype(np.float64).ravel()

        mask = (
            np.isfinite(latitude)
            & np.isfinite(longitude)
            & np.isfinite(xco2)
            & np.isfinite(u_met)
            & np.isfinite(v_met)
        )
        if quality_filter and "xco2_quality_flag" in handle:
            qflag = np.array(handle["xco2_quality_flag"][:]).astype(np.int32).ravel()
            mask &= qflag == 0

    return {
        "latitude": latitude[mask],
        "longitude": longitude[mask],
        "xco2": xco2[mask],
        "u": u_met[mask],
        "v": v_met[mask],
        "time": time_vals[mask],
    }


def grid_three_channels(points, lat0, lon0, grid_size=32, resolution_km=2.0, radius_factor=0.66):
    grid_x_km = make_grid_centers(grid_size, resolution_km)
    grid_y_km = make_grid_centers(grid_size, resolution_km)
    radius_km = radius_factor * resolution_km

    x_obs_km, y_obs_km = to_local_km(points["latitude"], points["longitude"], lat0, lon0)
    xco2_grid = grid_weighted_idw(
        x_obs_km, y_obs_km, points["xco2"], grid_x_km, grid_y_km, radius_km
    )
    u_grid = grid_weighted_idw(
        x_obs_km, y_obs_km, points["u"], grid_x_km, grid_y_km, radius_km
    )
    v_grid = grid_weighted_idw(
        x_obs_km, y_obs_km, points["v"], grid_x_km, grid_y_km, radius_km
    )
    valid_mask = np.isfinite(xco2_grid)
    return xco2_grid, u_grid, v_grid, valid_mask


def fill_per_channel_median(x):
    x = np.array(x, copy=True)
    for chan_idx in range(x.shape[-1]):
        chan = x[:, :, chan_idx]
        finite = chan[np.isfinite(chan)]
        fill_value = float(np.median(finite)) if finite.size else 0.0
        x[:, :, chan_idx] = np.where(np.isfinite(chan), chan, fill_value)
    return x.astype(np.float32)


def load_norm_spec(model_dir):
    path = Path(model_dir) / "norm_layer.keras"
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            weights_blob = zf.read("model.weights.h5")
        fd, tmp_path = tempfile.mkstemp(suffix=".h5")
        try:
            with open(tmp_path, "wb") as fout:
                fout.write(weights_blob)
            with h5py.File(tmp_path, "r") as handle:
                mean = np.array(
                    handle["_layer_checkpoint_dependencies/training_time_normalization/vars/0"][:],
                    dtype=np.float32,
                )
                variance = np.array(
                    handle["_layer_checkpoint_dependencies/training_time_normalization/vars/1"][:],
                    dtype=np.float32,
                )
            return {"type": "manual", "mean": mean, "variance": variance}
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
    model = tf.keras.models.load_model(
        path,
        custom_objects={"TrainingTimeNormalization": TrainingTimeNormalization},
    )
    return {"type": "model", "model": model}


def apply_norm(x_in, norm_spec):
    if norm_spec["type"] == "manual":
        mean = norm_spec["mean"].reshape((1, 1, 1, -1))
        variance = norm_spec["variance"].reshape((1, 1, 1, -1))
        return ((x_in - mean) / np.sqrt(variance)).astype(np.float32)
    return np.array(norm_spec["model"](x_in, training=True), dtype=np.float32)


def load_model(model_dir, weights_name):
    return tf.keras.models.load_model(Path(model_dir) / weights_name, compile=False)


def model_tag(model_dir):
    path = Path(model_dir)
    if path.parent.name == "inv_oco3_32_sweep":
        return "puresim_" + path.name
    return path.name


def load_cfg_fields(model_dir):
    cfg = OmegaConf.load(Path(model_dir) / "config.yaml")
    return {
        "loss_func": str(cfg.training.loss_func),
        "clouds_threshold": float(cfg.data.input.clouds_threshold),
        "chan_0": str(cfg.data.input.chan_0),
        "chan_1": str(cfg.data.input.chan_1),
        "chan_2": str(cfg.data.input.chan_2),
        "chan_3": str(getattr(cfg.data.input, "chan_3", "None")),
        "chan_4": str(getattr(cfg.data.input, "chan_4", "None")),
    }


def plot_comparison(df, out_path):
    pivot = df.pivot(index="obs_date", columns="model_name", values="predicted_emission")
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", linewidth=2, label=col)
    ax.set_ylabel("Predicted emission (Mt/yr)")
    ax.set_xlabel("Observation date")
    ax.set_title("Real Bełchatów OCO-3: pure-sim models vs mixed absolute")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare pure-sim author models against mixed absolute on real Bełchatów OCO-3."
    )
    parser.add_argument("--model-dirs", nargs="+", required=True)
    parser.add_argument("--mixed-csv", required=True)
    parser.add_argument("--nc4-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--weights-name", default="w_best.h5")
    parser.add_argument("--plant-lat", type=float, default=51.268)
    parser.add_argument("--plant-lon", type=float, default=19.331)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_files = [
        "oco3_LtCO2_210619_B10400Br_220318011522s.SUB.nc4",
        "oco3_LtCO2_220624_B10400Br_220929042413s.SUB.nc4",
        "oco3_LtCO2_221013_B10400Br_221205203141s.SUB.nc4",
        "oco3_LtCO2_230422_B10400Br_230611152400s.SUB.nc4",
    ]

    rows = []
    for model_dir in args.model_dirs:
        tag = model_tag(model_dir)
        cfg_fields = load_cfg_fields(model_dir)
        norm_spec = load_norm_spec(model_dir)
        model = load_model(model_dir, args.weights_name)

        for filename in selected_files:
            path = Path(args.nc4_dir) / filename
            points = load_raw_points(path, quality_filter=True)
            xco2_grid, u_grid, v_grid, valid_mask = grid_three_channels(
                points,
                lat0=args.plant_lat,
                lon0=args.plant_lon,
            )
            stacked = np.stack([xco2_grid, u_grid, v_grid], axis=-1)
            stacked = fill_per_channel_median(stacked)
            x_in = stacked[np.newaxis, ...]
            x_in = apply_norm(x_in, norm_spec)
            pred = float(np.squeeze(model.predict(x_in, verbose=0)))
            rows.append(
                {
                    "model_name": tag,
                    "model_dir": str(model_dir),
                    "obs_file": filename,
                    "obs_date": filename.split("_")[2],
                    "predicted_emission": pred,
                    "valid_pixels": int(valid_mask.sum()),
                    **cfg_fields,
                }
            )

    pure_df = pd.DataFrame(rows)

    mixed_df = pd.read_csv(args.mixed_csv).copy()
    mixed_df["model_name"] = "mixed_absolute"
    mixed_df["model_dir"] = str(Path(args.mixed_csv).parent)
    mixed_df["obs_file"] = mixed_df["name"].str.replace("_grid32", ".nc4", regex=False)
    mixed_df["obs_date"] = mixed_df["name"].str.extract(r"oco3_LtCO2_(\d{6})_")
    mixed_df["loss_func"] = "MeanAbsoluteError"
    mixed_df["clouds_threshold"] = 0.5
    mixed_df["chan_0"] = "xco2"
    mixed_df["chan_1"] = "None"
    mixed_df["chan_2"] = "None"
    if "predicted_emission" not in mixed_df.columns and "prediction" in mixed_df.columns:
        mixed_df = mixed_df.rename(columns={"prediction": "predicted_emission"})
    mixed_df = mixed_df[
        [
            "model_name",
            "model_dir",
            "obs_file",
            "obs_date",
            "predicted_emission",
            "valid_pixels",
            "loss_func",
            "clouds_threshold",
            "chan_0",
            "chan_1",
            "chan_2",
        ]
    ]

    combined = pd.concat([pure_df, mixed_df], ignore_index=True)
    combined.to_csv(out_dir / "puresim_vs_mixed_absolute.csv", index=False)
    plot_comparison(combined, out_dir / "puresim_vs_mixed_absolute.png")

    print("Saved:", out_dir / "puresim_vs_mixed_absolute.csv")
    print("Saved:", out_dir / "puresim_vs_mixed_absolute.png")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
