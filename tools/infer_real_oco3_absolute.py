import argparse
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.preprocessing import TrainingTimeNormalization

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_norm_model(model_dir: Path):
    return tf.keras.models.load_model(
        model_dir / "norm_layer.keras",
        custom_objects={"TrainingTimeNormalization": TrainingTimeNormalization},
    )


def load_model(model_dir: Path, weights_name: str):
    config_path = model_dir / "config.yaml"
    if config_path.exists():
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        if str(cfg.model.name) in ["hierarchical_plume_reasoner_tiny"]:
            from models.reg import Reg_model_builder

            model = Reg_model_builder(
                name=str(cfg.model.name),
                input_shape=[32, 32, 3],
                classes=int(getattr(cfg.data.output, "N_emissions", 1)),
                dropout_rate=float(getattr(cfg.model, "dropout_rate", 0.2)),
            ).get_model()
            model.load_weights(model_dir / weights_name)
            return model
    return tf.keras.models.load_model(model_dir / weights_name, compile=False)


def select_emission_output(pred):
    if isinstance(pred, dict):
        return pred.get("emission", next(iter(pred.values())))
    if isinstance(pred, (list, tuple)):
        return pred[0]
    return pred


def compute_fill_values(train_h5: Path, field_name: str) -> np.ndarray:
    with h5py.File(train_h5, "r") as handle:
        array = np.array(handle[field_name][:], dtype=np.float32)
    if array.ndim == 3:
        array = array[..., np.newaxis]
    fill_values = np.nanmedian(array, axis=(0, 1, 2))
    fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0)
    return fill_values.astype(np.float32)


def km_per_deg_lon(lat_deg: float) -> float:
    return 111.32 * np.cos(np.deg2rad(lat_deg))


def to_local_km(latitude, longitude, lat0, lon0):
    x_km = (longitude - lon0) * km_per_deg_lon(lat0)
    y_km = (latitude - lat0) * 110.57
    return x_km, y_km


def make_grid_centers(size: int, resolution_km: float):
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


def load_raw_points(nc4_path: Path, quality_filter: bool = True):
    with h5py.File(nc4_path, "r") as handle:
        latitude = np.array(handle["latitude"][:], dtype=np.float32).ravel()
        longitude = np.array(handle["longitude"][:], dtype=np.float32).ravel()
        xco2 = np.array(handle["xco2"][:], dtype=np.float32).ravel()
        u_met = np.array(
            handle["Meteorology"]["windspeed_u_met"][:],
            dtype=np.float32,
        ).ravel()
        v_met = np.array(
            handle["Meteorology"]["windspeed_v_met"][:],
            dtype=np.float32,
        ).ravel()

        mask = (
            np.isfinite(latitude)
            & np.isfinite(longitude)
            & np.isfinite(xco2)
            & np.isfinite(u_met)
            & np.isfinite(v_met)
        )
        if quality_filter and "xco2_quality_flag" in handle:
            qflag = np.array(handle["xco2_quality_flag"][:], dtype=np.int32).ravel()
            mask &= qflag == 0

    return {
        "latitude": latitude[mask],
        "longitude": longitude[mask],
        "xco2": xco2[mask],
        "u": u_met[mask],
        "v": v_met[mask],
    }


def build_grid32_from_nc4(
    nc4_path: Path,
    plant_lat: float,
    plant_lon: float,
    grid_size: int = 32,
    resolution_km: float = 2.0,
    radius_factor: float = 0.66,
):
    points = load_raw_points(nc4_path, quality_filter=True)
    grid_x_km = make_grid_centers(grid_size, resolution_km)
    grid_y_km = make_grid_centers(grid_size, resolution_km)
    radius_km = radius_factor * resolution_km
    x_obs_km, y_obs_km = to_local_km(
        points["latitude"],
        points["longitude"],
        plant_lat,
        plant_lon,
    )
    xco2_grid = grid_weighted_idw(
        x_obs_km, y_obs_km, points["xco2"], grid_x_km, grid_y_km, radius_km
    )
    u_grid = grid_weighted_idw(
        x_obs_km, y_obs_km, points["u"], grid_x_km, grid_y_km, radius_km
    )
    v_grid = grid_weighted_idw(
        x_obs_km, y_obs_km, points["v"], grid_x_km, grid_y_km, radius_km
    )
    neighbor_count = np.zeros_like(xco2_grid, dtype=np.int32)
    for iy, y0 in enumerate(grid_y_km):
        dy = y_obs_km - y0
        for ix, x0 in enumerate(grid_x_km):
            dx = x_obs_km - x0
            dist = np.sqrt(dx * dx + dy * dy)
            neighbor_count[iy, ix] = int(np.sum(dist < radius_km))
    return {
        "xco2_grid": xco2_grid,
        "u_grid": u_grid,
        "v_grid": v_grid,
        "valid_mask": np.isfinite(xco2_grid),
        "neighbor_count": neighbor_count,
        "grid_x_km": grid_x_km,
        "grid_y_km": grid_y_km,
        "plant_lat": plant_lat,
        "plant_lon": plant_lon,
    }


def fill_per_channel(array: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    filled = np.array(array, copy=True, dtype=np.float32)
    for chan_idx in range(filled.shape[-1]):
        filled[:, :, chan_idx] = np.where(
            np.isfinite(filled[:, :, chan_idx]),
            filled[:, :, chan_idx],
            fill_values[chan_idx],
        )
    return filled


def masked_gaussian_lowpass(
    background: np.ndarray,
    mask: np.ndarray,
    sigma_px: float = 3.0,
    min_weight: float = 1e-3,
):
    mask_f = mask.astype(np.float32)
    background_filled = np.where(mask, background, 0.0).astype(np.float32)
    numerator = gaussian_filter(background_filled * mask_f, sigma=sigma_px, mode="constant", cval=0.0)
    denominator = gaussian_filter(mask_f, sigma=sigma_px, mode="constant", cval=0.0)
    low = np.full_like(background_filled, np.nan, dtype=np.float32)
    valid = denominator > min_weight
    low[valid] = numerator[valid] / denominator[valid]
    return low.astype(np.float32)


def prepare_input_from_nc4(
    nc4_path: Path,
    field_name: str,
    fill_values: np.ndarray,
    norm_model,
    plant_lat: float,
    plant_lon: float,
):
    gridded = build_grid32_from_nc4(nc4_path, plant_lat=plant_lat, plant_lon=plant_lon)
    xco2_base = gridded["xco2_grid"]
    if "xco2_mixed" in field_name and "absolute" not in field_name:
        xco2_low = masked_gaussian_lowpass(
            xco2_base,
            np.isfinite(xco2_base),
            sigma_px=3.0,
        )
        xco2_base = np.where(np.isfinite(xco2_base), xco2_base - xco2_low, np.nan).astype(np.float32)

    if field_name.endswith("_uv_flatwind"):
        u_mean = np.nanmean(gridded["u_grid"])
        v_mean = np.nanmean(gridded["v_grid"])
        u_field = np.full_like(gridded["u_grid"], u_mean, dtype=np.float32)
        v_field = np.full_like(gridded["v_grid"], v_mean, dtype=np.float32)
        stacked = np.stack(
            [
                xco2_base,
                u_field,
                v_field,
            ],
            axis=-1,
        )
    elif field_name.endswith("_uv"):
        stacked = np.stack(
            [
                xco2_base,
                gridded["u_grid"],
                gridded["v_grid"],
            ],
            axis=-1,
        )
    else:
        stacked = xco2_base[..., np.newaxis]
    stacked = fill_per_channel(stacked, fill_values)
    x_norm = np.array(norm_model(stacked[np.newaxis, ...], training=True), dtype=np.float32)
    return {
        "x_raw": stacked,
        "x_norm": x_norm,
        **gridded,
    }


def make_summary_plot(rows: List[Dict], out_path: Path, field_name: str):
    n = len(rows)
    ncols = 2 if not field_name.endswith("_uv") else 4
    fig, axes = plt.subplots(nrows=n, ncols=ncols, figsize=(5 * ncols, 4 * n), squeeze=False)
    for i, row in enumerate(rows):
        raw_xco2 = np.where(row["valid_mask"], row["x_raw"][:, :, 0], np.nan)
        cmap0 = plt.get_cmap("viridis").copy()
        cmap0.set_bad(color="white")
        im0 = axes[i, 0].imshow(raw_xco2, origin="lower", cmap=cmap0)
        axes[i, 0].set_title(f"{row['name']}\nRaw xco2 | pred={row['prediction']:.3f}")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        norm_xco2 = row["x_norm"][0, :, :, 0]
        cmap1 = plt.get_cmap("coolwarm").copy()
        cmap1.set_bad(color="white")
        lim = float(np.nanpercentile(np.abs(norm_xco2), 99)) if np.isfinite(norm_xco2).any() else 1.0
        lim = max(lim, 1.0)
        im1 = axes[i, 1].imshow(norm_xco2, origin="lower", cmap=cmap1, vmin=-lim, vmax=lim)
        axes[i, 1].set_title(f"Normalized xco2 | finite={int(row['valid_mask'].sum())}/1024")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        if field_name.endswith("_uv"):
            for j, title in [(2, "u_wind"), (3, "v_wind")]:
                raw_chan = row["x_raw"][:, :, j - 1]
                lim = float(np.nanpercentile(np.abs(raw_chan), 99)) if np.isfinite(raw_chan).any() else 1.0
                lim = max(lim, 1.0)
                im = axes[i, j].imshow(raw_chan, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
                axes[i, j].set_title(title)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run mixed-model inference on real OCO-3 nc4 files."
    )
    parser.add_argument("--model-dir", required=True, help="Trained model directory.")
    parser.add_argument("--train-h5", required=True, help="Mixed train.h5 used to compute NaN fill values.")
    parser.add_argument("--nc4-dir", required=True, help="Directory containing source .nc4 files.")
    parser.add_argument("--weights-name", default="w_best.h5", help="Weights file to load.")
    parser.add_argument("--field-name", default="xco2_mixed_absolute", help="Training field used for fill statistics.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--plant-lat", type=float, default=51.268)
    parser.add_argument("--plant-lon", type=float, default=19.331)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    train_h5 = Path(args.train_h5)
    nc4_dir = Path(args.nc4_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_files = [
        "oco3_LtCO2_210619_B10400Br_220318011522s.SUB.nc4",
        "oco3_LtCO2_220624_B10400Br_220929042413s.SUB.nc4",
        "oco3_LtCO2_221013_B10400Br_221205203141s.SUB.nc4",
        "oco3_LtCO2_230422_B10400Br_230611152400s.SUB.nc4",
    ]

    fill_values = compute_fill_values(train_h5, args.field_name)
    norm_model = load_norm_model(model_dir)
    model = load_model(model_dir, args.weights_name)

    rows = []
    for filename in selected_files:
        prepared = prepare_input_from_nc4(
            nc4_dir / filename,
            field_name=args.field_name,
            fill_values=fill_values,
            norm_model=norm_model,
            plant_lat=args.plant_lat,
            plant_lon=args.plant_lon,
        )
        pred = float(np.squeeze(select_emission_output(model.predict(prepared["x_norm"], verbose=0))))
        rows.append(
            {
                "name": filename.replace(".nc4", ""),
                "prediction": pred,
                "valid_pixels": int(prepared["valid_mask"].sum()),
                "missing_pixels": int((~prepared["valid_mask"]).sum()),
                "xco2_min": float(np.nanmin(prepared["x_raw"][:, :, 0])),
                "xco2_max": float(np.nanmax(prepared["x_raw"][:, :, 0])),
                "plant_lat": prepared["plant_lat"],
                "plant_lon": prepared["plant_lon"],
                **prepared,
            }
        )

    summary_df = pd.DataFrame(
        [
            {
                "name": row["name"],
                "prediction": row["prediction"],
                "predicted_emission": row["prediction"],
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
    summary_df.to_csv(out_dir / "oco3_absolute_predictions.csv", index=False)
    make_summary_plot(rows, out_dir / "oco3_absolute_predictions.png", args.field_name)

    print("Saved predictions to:", out_dir / "oco3_absolute_predictions.csv")
    print("Saved preview plot to:", out_dir / "oco3_absolute_predictions.png")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
