from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def km_per_deg_lon(lat_deg: float) -> float:
    return 111.32 * np.cos(np.deg2rad(lat_deg))


def to_local_km(
    latitude: np.ndarray, longitude: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    x_km = (longitude - lon0) * km_per_deg_lon(lat0)
    y_km = (latitude - lat0) * 110.57
    return x_km, y_km


def to_lon_lat(
    x_km: np.ndarray, y_km: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    lon = lon0 + x_km / km_per_deg_lon(lat0)
    lat = lat0 + y_km / 110.57
    return lon, lat


def make_grid_centers(size: int, resolution_km: float) -> np.ndarray:
    half_span = size * resolution_km / 2.0
    return np.linspace(
        -half_span + resolution_km / 2.0,
        half_span - resolution_km / 2.0,
        size,
    )


def grid_weighted_idw(
    x_obs_km: np.ndarray,
    y_obs_km: np.ndarray,
    values: np.ndarray,
    grid_x_km: np.ndarray,
    grid_y_km: np.ndarray,
    radius_km: float,
    power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((grid_y_km.size, grid_x_km.size), np.nan, dtype=float)
    counts = np.zeros_like(grid, dtype=int)

    for iy, y0 in enumerate(grid_y_km):
        dy = y_obs_km - y0
        for ix, x0 in enumerate(grid_x_km):
            dx = x_obs_km - x0
            dist = np.sqrt(dx * dx + dy * dy)
            neighborhood = dist < radius_km
            n_neighbors = int(np.count_nonzero(neighborhood))
            counts[iy, ix] = n_neighbors
            if n_neighbors == 0:
                continue

            local_dist = dist[neighborhood]
            local_values = values[neighborhood]
            if np.any(local_dist < 1e-6):
                grid[iy, ix] = float(local_values[np.argmin(local_dist)])
                continue

            weights = 1.0 / np.power(local_dist, power)
            grid[iy, ix] = float(np.sum(weights * local_values) / np.sum(weights))

    return grid, counts


def masked_gaussian_lowpass(grid: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(grid)
    filled = np.where(valid, grid, 0.0)
    num = gaussian_filter(filled * valid.astype(float), sigma=sigma_px, mode="nearest")
    den = gaussian_filter(valid.astype(float), sigma=sigma_px, mode="nearest")
    low = np.full_like(grid, np.nan, dtype=float)
    ok = den > 1e-6
    low[ok] = num[ok] / den[ok]
    return low, den


def center_bias_ppm(residual: np.ndarray) -> float:
    valid = np.isfinite(residual)
    if not np.any(valid):
        return float("nan")
    n = residual.shape[0]
    yy, xx = np.indices(residual.shape)
    cy = (n - 1) / 2.0
    cx = (n - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    core = (rr <= 2.5) & valid
    ring = (rr >= 5.0) & (rr <= 8.0) & valid
    if np.count_nonzero(core) == 0 or np.count_nonzero(ring) == 0:
        return float("nan")
    return float(np.nanmean(residual[core]) - np.nanmean(residual[ring]))


def positive_tail_ratio(residual: np.ndarray) -> float:
    valid = np.isfinite(residual)
    if not np.any(valid):
        return float("nan")
    sigma = float(np.nanstd(residual))
    if sigma <= 0:
        return 0.0
    threshold = max(1.0, 1.5 * sigma)
    return float(np.count_nonzero((residual > threshold) & valid) / np.count_nonzero(valid))


def anisotropy_ratio(residual: np.ndarray) -> float:
    valid = np.isfinite(residual)
    if np.count_nonzero(valid) < 20:
        return float("nan")
    yy, xx = np.indices(residual.shape)
    vals = residual[valid]
    weights = np.abs(vals - np.nanmean(vals)) + 1e-6
    x = xx[valid].astype(float)
    y = yy[valid].astype(float)
    x0 = np.average(x, weights=weights)
    y0 = np.average(y, weights=weights)
    dx = x - x0
    dy = y - y0
    cov_xx = np.average(dx * dx, weights=weights)
    cov_yy = np.average(dy * dy, weights=weights)
    cov_xy = np.average(dx * dy, weights=weights)
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals[0] <= 1e-8:
        return float("inf")
    return float(np.sqrt(eigvals[-1] / eigvals[0]))


def background_like_judgement(
    valid_fraction: float,
    center_bias: float,
    tail_ratio: float,
    residual_std: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if valid_fraction < 0.60:
        reasons.append("valid_fraction too low")
    if np.isfinite(center_bias) and abs(center_bias) >= 0.50:
        reasons.append("center bias too strong")
    if np.isfinite(tail_ratio) and tail_ratio >= 0.03:
        reasons.append("positive tail too strong")
    if np.isfinite(residual_std) and residual_std >= 1.00:
        reasons.append("residual variance too large")
    if reasons:
        return "not_background_like", reasons
    return "background_like", ["weak center anomaly", "limited positive tail", "acceptable coverage"]


def parse_time(text: str) -> float:
    dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a local SAM scene from an OCO-3 daily Lite file.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--start-utc", required=True, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--end-utc", required=True, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--resolution-km", type=float, default=2.0)
    parser.add_argument("--radius-factor", type=float, default=0.66)
    parser.add_argument("--sigma-px", type=float, default=3.0)
    parser.add_argument("--bbox-lon-min", type=float, default=None)
    parser.add_argument("--bbox-lon-max", type=float, default=None)
    parser.add_argument("--bbox-lat-min", type=float, default=None)
    parser.add_argument("--bbox-lat-max", type=float, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ts = parse_time(args.start_utc)
    end_ts = parse_time(args.end_utc)

    with h5py.File(args.input_file, "r") as handle:
        time = np.array(handle["time"]).astype(float).ravel()
        latitude = np.array(handle["latitude"]).astype(float).ravel()
        longitude = np.array(handle["longitude"]).astype(float).ravel()
        xco2 = np.array(handle["xco2"]).astype(float).ravel()
        qf = np.array(handle["xco2_quality_flag"]).astype(int).ravel()

    scene_mask = (
        (time >= start_ts)
        & (time <= end_ts)
        & np.isfinite(latitude)
        & np.isfinite(longitude)
        & np.isfinite(xco2)
    )
    if args.bbox_lon_min is not None:
        scene_mask &= longitude >= args.bbox_lon_min
    if args.bbox_lon_max is not None:
        scene_mask &= longitude <= args.bbox_lon_max
    if args.bbox_lat_min is not None:
        scene_mask &= latitude >= args.bbox_lat_min
    if args.bbox_lat_max is not None:
        scene_mask &= latitude <= args.bbox_lat_max
    scene_all = int(np.count_nonzero(scene_mask))
    if scene_all == 0:
        raise SystemExit("No soundings found in the requested time window.")

    qf0_mask = scene_mask & (qf == 0)
    if np.count_nonzero(qf0_mask) == 0:
        raise SystemExit("No qf=0 soundings found in the requested time window.")

    latitude_sel = latitude[qf0_mask]
    longitude_sel = longitude[qf0_mask]
    xco2_sel = xco2[qf0_mask]
    time_sel = time[qf0_mask]

    lat0 = float(np.median(latitude_sel))
    lon0 = float(np.median(longitude_sel))

    grid_x_km = make_grid_centers(args.grid_size, args.resolution_km)
    grid_y_km = make_grid_centers(args.grid_size, args.resolution_km)
    radius_km = args.radius_factor * args.resolution_km

    x_obs_km, y_obs_km = to_local_km(latitude_sel, longitude_sel, lat0, lon0)
    grid, counts = grid_weighted_idw(
        x_obs_km=x_obs_km,
        y_obs_km=y_obs_km,
        values=xco2_sel,
        grid_x_km=grid_x_km,
        grid_y_km=grid_y_km,
        radius_km=radius_km,
    )
    background_low, low_weight = masked_gaussian_lowpass(grid, sigma_px=args.sigma_px)
    residual = grid - background_low

    valid_fraction = float(np.count_nonzero(np.isfinite(grid)) / grid.size)
    residual_std = float(np.nanstd(residual))
    c_bias = center_bias_ppm(residual)
    tail_ratio = positive_tail_ratio(residual)
    aniso = anisotropy_ratio(residual)
    judgement, reasons = background_like_judgement(valid_fraction, c_bias, tail_ratio, residual_std)

    stem = args.scene_name
    np.savez_compressed(
        out_dir / f"{stem}_scene.npz",
        latitude=latitude_sel,
        longitude=longitude_sel,
        xco2=xco2_sel,
        time=time_sel,
        center_lat=lat0,
        center_lon=lon0,
        xco2_grid=grid,
        background_low=background_low,
        background_residual=residual,
        neighbor_count=counts,
        lowpass_weight=low_weight,
        grid_x_km=grid_x_km,
        grid_y_km=grid_y_km,
    )

    metadata = {
        "scene_name": stem,
        "input_file": str(Path(args.input_file).resolve()),
        "start_utc": args.start_utc,
        "end_utc": args.end_utc,
        "bbox_lon_min": args.bbox_lon_min,
        "bbox_lon_max": args.bbox_lon_max,
        "bbox_lat_min": args.bbox_lat_min,
        "bbox_lat_max": args.bbox_lat_max,
        "soundings_all_in_window": scene_all,
        "soundings_qf0_in_window": int(np.count_nonzero(qf0_mask)),
        "center_lat": lat0,
        "center_lon": lon0,
        "time_min_utc": datetime.fromtimestamp(float(time_sel.min()), tz=timezone.utc).isoformat(),
        "time_max_utc": datetime.fromtimestamp(float(time_sel.max()), tz=timezone.utc).isoformat(),
        "valid_fraction": valid_fraction,
        "residual_std_ppm": residual_std,
        "center_bias_ppm": c_bias,
        "positive_tail_ratio": tail_ratio,
        "anisotropy_ratio": aniso,
        "judgement": judgement,
        "reasons": reasons,
    }
    (out_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lon_edges, lat_edges = to_lon_lat(
        np.array([grid_x_km[0] - args.resolution_km / 2.0, grid_x_km[-1] + args.resolution_km / 2.0]),
        np.array([grid_y_km[0] - args.resolution_km / 2.0, grid_y_km[-1] + args.resolution_km / 2.0]),
        lat0,
        lon0,
    )
    extent = [lon_edges[0], lon_edges[1], lat_edges[0], lat_edges[1]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    axes[0].scatter(longitude_sel, latitude_sel, c=xco2_sel, s=7, cmap="viridis")
    axes[0].set_title(f"{stem}\nRaw qf=0 soundings")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    cmap_abs = plt.get_cmap("viridis").copy()
    cmap_abs.set_bad("white")
    im1 = axes[1].imshow(grid, origin="lower", extent=extent, interpolation="nearest", cmap=cmap_abs)
    axes[1].set_title("32x32 gridded XCO2")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    div1 = make_axes_locatable(axes[1])
    cax1 = div1.append_axes("right", size="4.5%", pad=0.08)
    plt.colorbar(im1, cax=cax1)

    cmap_res = plt.get_cmap("coolwarm").copy()
    cmap_res.set_bad("white")
    vmax = np.nanpercentile(np.abs(residual), 99) if np.any(np.isfinite(residual)) else 1.0
    vmax = max(float(vmax), 0.5)
    im2 = axes[2].imshow(
        residual,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=cmap_res,
        vmin=-vmax,
        vmax=vmax,
    )
    axes[2].set_title(f"Residual\n{judgement}")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    div2 = make_axes_locatable(axes[2])
    cax2 = div2.append_axes("right", size="4.5%", pad=0.08)
    plt.colorbar(im2, cax=cax2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
