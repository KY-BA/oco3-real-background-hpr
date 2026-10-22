import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


SCENE_CATEGORY = {
    "scene_20210124_142615_142820_lat13p2_13p78_lon2p35_3p25": "arid",
    "scene_20210204_182755_183000_lat36p42_37p00_lonm98p4_m97p5": "plain",
    "scene_20210420_210835_211040_lat38p25_38p83_lonm122p0_m121p1": "urban",
    "scene_20220725_010105_010310_lat43p22_43p80_lonm120p2_m119p3": "arid",
    "scene_20221017_195555_195800_lat39p22_39p80_lonm97p3_m96p4": "plain",
    "scene_20221020_033705_033910_lat43p75_44p33_lon124p3_125p2": "forest",
    "scene_20231006_192245_192450_lat43p20_43p78_lonm120p3_m119p4": "arid",
    "scene_20250927_200015_200220_lat39p62_40p20_lonm104p5_m103p6": "mountain",
    "scene_20251212_041005_041210_lat45p75_46p33_lon125p6_126p5": "forest",
    "scene_20200814_204505_204710_lat44p22_44p80_lonm89p8_m88p9": "plain",
    "scene_20210421_074525_074725_lat48p30_48p88_lon1p9_2p8": "urban",
    "scene_20210614_105355_105600_lat50p1_50p68_lon5_5p9": "forest",
    "scene_20211130_055145_055345_lat25p9_26p48_lon105p9_106p8": "mountain",
}

TARGET_PER_CATEGORY_TRAIN = 30072
DEFAULT_MID_FRAC = 0.20
DEFAULT_HIGH_FRAC = 0.10


def decode_strings(values) -> np.ndarray:
    out = []
    for item in values:
        if isinstance(item, (bytes, bytearray)):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return np.array(out, dtype=object)


@dataclass
class BackgroundScene:
    path: Path
    library: str
    name: str
    category: str
    xco2_grid: np.ndarray
    mask: np.ndarray
    neighbor_count: np.ndarray


@dataclass
class AugmentedRow:
    plume_idx: int
    alpha: float
    row_type: str


def load_background_scene(path: Path, library: str) -> BackgroundScene:
    data = np.load(path)
    name = path.stem.replace("_scene", "")
    if name not in SCENE_CATEGORY:
        raise KeyError(f"Missing category mapping for background: {name}")
    xco2_grid = data["xco2_grid"].astype(np.float32)
    neighbor_count = data["neighbor_count"].astype(np.int32)
    mask = np.isfinite(xco2_grid)
    return BackgroundScene(
        path=path,
        library=library,
        name=name,
        category=SCENE_CATEGORY[name],
        xco2_grid=xco2_grid,
        mask=mask,
        neighbor_count=neighbor_count,
    )


def list_background_scenes(a_dir: Path, b_dir: Path) -> Dict[str, BackgroundScene]:
    scenes: Dict[str, BackgroundScene] = {}
    for directory, library in [(a_dir, "A_class"), (b_dir, "B_class")]:
        for path in sorted(directory.glob("*_scene.npz")):
            scene = load_background_scene(path, library)
            scenes[scene.name] = scene
    return scenes


def load_split_map_from_manifest(
    manifest_path: Path, scene_lookup: Dict[str, BackgroundScene]
) -> Dict[str, List[BackgroundScene]]:
    manifest = pd.read_csv(manifest_path)
    split_map: Dict[str, List[BackgroundScene]] = {}
    for split_name, group in manifest.groupby("split", sort=False):
        names = group["background_name"].tolist()
        split_map[split_name] = [scene_lookup[name] for name in names]
    return split_map


def load_plume_library(dataset_path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(dataset_path, "r") as handle:
        plume = np.array(handle["plume"][:], dtype=np.float32)
        emiss_all = np.array(handle["emiss"][:], dtype=np.float32)
        source_idx_img = np.array(handle["idx_img"][:], dtype=np.int32)
        point_source = decode_strings(handle["point_source"][:])

    return {
        "plume": plume,
        "emiss_all": emiss_all,
        "emiss_target": emiss_all[:, 1].astype(np.float32),
        "source_idx_img": source_idx_img,
        "point_source": point_source,
    }


def indices_by_point_source(point_source: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: np.where(point_source == name)[0] for name in np.unique(point_source)}


def build_balanced_train_plan(
    scenes: Sequence[BackgroundScene],
    plume_lib: Dict[str, np.ndarray],
    seed: int,
) -> Dict[str, np.ndarray]:
    by_category: Dict[str, List[BackgroundScene]] = {}
    for scene in scenes:
        by_category.setdefault(scene.category, []).append(scene)

    source_map = indices_by_point_source(plume_lib["point_source"])
    rng = np.random.default_rng(seed)
    plan: Dict[str, np.ndarray] = {}

    for category, category_scenes in by_category.items():
        n_scenes = len(category_scenes)
        divisor = len(source_map) * n_scenes
        if TARGET_PER_CATEGORY_TRAIN % divisor != 0:
            raise ValueError(f"Cannot evenly balance category {category} with {n_scenes} scenes.")
        n_per_source_per_scene = TARGET_PER_CATEGORY_TRAIN // divisor
        for scene in category_scenes:
            parts = []
            for _, source_idx in source_map.items():
                chosen = np.sort(rng.choice(source_idx, size=n_per_source_per_scene, replace=False))
                parts.append(chosen)
            plan[scene.name] = np.concatenate(parts)
    return plan


def full_plan(scenes: Sequence[BackgroundScene], n_samples: int) -> Dict[str, np.ndarray]:
    idx = np.arange(n_samples, dtype=np.int32)
    return {scene.name: idx for scene in scenes}


def sample_alpha_for_range(
    emiss: np.ndarray,
    rng: np.random.Generator,
    target_low: float,
    target_high: float,
    alpha_min: float,
    alpha_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    low_eff = np.maximum(target_low, alpha_min * emiss)
    high_eff = np.minimum(target_high, alpha_max * emiss)
    eligible = np.where(low_eff <= high_eff)[0]
    return eligible, np.stack([low_eff, high_eff], axis=1)


def build_alpha_plan(
    scenes: Sequence[BackgroundScene],
    base_plan: Dict[str, np.ndarray],
    plume_lib: Dict[str, np.ndarray],
    seed: int,
    mid_frac: float,
    high_frac: float,
    alpha_min: float,
    alpha_max: float,
    mid_range: Tuple[float, float],
    high_range: Tuple[float, float],
) -> Dict[str, List[AugmentedRow]]:
    rng = np.random.default_rng(seed)
    emiss_target = plume_lib["emiss_target"]

    by_category: Dict[str, List[BackgroundScene]] = {}
    for scene in scenes:
        by_category.setdefault(scene.category, []).append(scene)

    alpha_plan: Dict[str, List[AugmentedRow]] = {scene.name: [] for scene in scenes}

    for category, category_scenes in by_category.items():
        pooled_scene_names: List[str] = []
        pooled_indices: List[int] = []
        for scene in category_scenes:
            selected = base_plan[scene.name]
            pooled_scene_names.extend([scene.name] * len(selected))
            pooled_indices.extend(selected.tolist())
        pooled_indices_arr = np.array(pooled_indices, dtype=np.int32)
        pooled_scene_names_arr = np.array(pooled_scene_names, dtype=object)
        pooled_emiss = emiss_target[pooled_indices_arr]

        n_mid = int(round(TARGET_PER_CATEGORY_TRAIN * mid_frac))
        n_high = int(round(TARGET_PER_CATEGORY_TRAIN * high_frac))

        for row_type, n_target, target_range in [
            ("alpha_mid", n_mid, mid_range),
            ("alpha_high", n_high, high_range),
        ]:
            eligible_mask, low_high = sample_alpha_for_range(
                pooled_emiss,
                rng,
                target_low=target_range[0],
                target_high=target_range[1],
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )
            if eligible_mask.size == 0:
                raise ValueError(f"No eligible samples for {category} {row_type}.")
            chosen_positions = rng.choice(eligible_mask, size=n_target, replace=True)
            low = low_high[chosen_positions, 0]
            high = low_high[chosen_positions, 1]
            sampled_targets = rng.uniform(low, high).astype(np.float32)
            chosen_emiss = pooled_emiss[chosen_positions]
            alpha_vals = (sampled_targets / chosen_emiss).astype(np.float32)
            alpha_vals = np.clip(alpha_vals, alpha_min, alpha_max)

            for pos, alpha in zip(chosen_positions, alpha_vals):
                scene_name = pooled_scene_names_arr[pos]
                plume_idx = int(pooled_indices_arr[pos])
                alpha_plan[scene_name].append(
                    AugmentedRow(plume_idx=plume_idx, alpha=float(alpha), row_type=row_type)
                )

    return alpha_plan


def init_output_file(path: Path, total_samples: int, metadata: Dict[str, str]) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "xco2_mixed_absolute",
            shape=(total_samples, 32, 32),
            dtype=np.float32,
            chunks=(1, 32, 32),
            compression="gzip",
        )
        handle.create_dataset(
            "background_mask",
            shape=(total_samples, 32, 32),
            dtype=np.uint8,
            chunks=(1, 32, 32),
            compression="gzip",
        )
        handle.create_dataset(
            "background_neighbor_count",
            shape=(total_samples, 32, 32),
            dtype=np.int32,
            chunks=(1, 32, 32),
            compression="gzip",
        )
        handle.create_dataset("emiss_all", shape=(total_samples, 3), dtype=np.float32, compression="gzip")
        handle.create_dataset("emiss_target", shape=(total_samples,), dtype=np.float32, compression="gzip")
        handle.create_dataset("source_idx_img", shape=(total_samples,), dtype=np.int32, compression="gzip")
        handle.create_dataset("point_source", shape=(total_samples,), dtype=string_dtype)
        handle.create_dataset("background_name", shape=(total_samples,), dtype=string_dtype)
        handle.create_dataset("background_library", shape=(total_samples,), dtype=string_dtype)
        handle.create_dataset("background_category", shape=(total_samples,), dtype=string_dtype)
        handle.create_dataset("alpha_multiplier", shape=(total_samples,), dtype=np.float32, compression="gzip")
        handle.create_dataset("augmented_flag", shape=(total_samples,), dtype=np.uint8, compression="gzip")
        handle.create_dataset("source_row_type", shape=(total_samples,), dtype=string_dtype)
        for key, value in metadata.items():
            handle.attrs[key] = value


def write_split(
    out_path: Path,
    scenes: Sequence[BackgroundScene],
    plume_lib: Dict[str, np.ndarray],
    metadata: Dict[str, str],
    plan: Dict[str, np.ndarray],
    alpha_plan: Optional[Dict[str, List[AugmentedRow]]] = None,
) -> None:
    alpha_plan = alpha_plan or {scene.name: [] for scene in scenes}
    total_samples = int(sum(len(plan[scene.name]) + len(alpha_plan.get(scene.name, [])) for scene in scenes))
    init_output_file(out_path, total_samples, metadata)

    plume = plume_lib["plume"]
    emiss_all = plume_lib["emiss_all"]
    emiss_target = plume_lib["emiss_target"]
    source_idx_img = plume_lib["source_idx_img"]
    point_source = plume_lib["point_source"].astype(object)

    with h5py.File(out_path, "a") as handle:
        cursor = 0
        for scene in scenes:
            base_selected = plan[scene.name]
            base_n = len(base_selected)
            aug_rows = alpha_plan.get(scene.name, [])
            aug_n = len(aug_rows)
            next_cursor = cursor + base_n + aug_n

            if base_n > 0:
                base_absolute = np.where(
                    scene.mask[None, :, :],
                    scene.xco2_grid[None, :, :] + plume[base_selected],
                    np.nan,
                ).astype(np.float32)

                handle["xco2_mixed_absolute"][cursor:cursor + base_n] = base_absolute
                handle["background_mask"][cursor:cursor + base_n] = np.broadcast_to(
                    scene.mask.astype(np.uint8)[None, :, :], (base_n, 32, 32)
                )
                handle["background_neighbor_count"][cursor:cursor + base_n] = np.broadcast_to(
                    scene.neighbor_count[None, :, :], (base_n, 32, 32)
                )
                handle["emiss_all"][cursor:cursor + base_n] = emiss_all[base_selected]
                handle["emiss_target"][cursor:cursor + base_n] = emiss_target[base_selected]
                handle["source_idx_img"][cursor:cursor + base_n] = source_idx_img[base_selected]
                handle["point_source"][cursor:cursor + base_n] = point_source[base_selected]
                handle["background_name"][cursor:cursor + base_n] = np.repeat(scene.name, base_n)
                handle["background_library"][cursor:cursor + base_n] = np.repeat(scene.library, base_n)
                handle["background_category"][cursor:cursor + base_n] = np.repeat(scene.category, base_n)
                handle["alpha_multiplier"][cursor:cursor + base_n] = 1.0
                handle["augmented_flag"][cursor:cursor + base_n] = 0
                handle["source_row_type"][cursor:cursor + base_n] = np.repeat("base", base_n)

            if aug_n > 0:
                aug_indices = np.array([row.plume_idx for row in aug_rows], dtype=np.int32)
                aug_alpha = np.array([row.alpha for row in aug_rows], dtype=np.float32)
                aug_types = np.array([row.row_type for row in aug_rows], dtype=object)
                scaled_plume = plume[aug_indices] * aug_alpha[:, None, None]
                aug_absolute = np.where(
                    scene.mask[None, :, :],
                    scene.xco2_grid[None, :, :] + scaled_plume,
                    np.nan,
                ).astype(np.float32)
                start = cursor + base_n
                end = start + aug_n
                handle["xco2_mixed_absolute"][start:end] = aug_absolute
                handle["background_mask"][start:end] = np.broadcast_to(
                    scene.mask.astype(np.uint8)[None, :, :], (aug_n, 32, 32)
                )
                handle["background_neighbor_count"][start:end] = np.broadcast_to(
                    scene.neighbor_count[None, :, :], (aug_n, 32, 32)
                )
                handle["emiss_all"][start:end] = emiss_all[aug_indices] * aug_alpha[:, None]
                handle["emiss_target"][start:end] = emiss_target[aug_indices] * aug_alpha
                handle["source_idx_img"][start:end] = source_idx_img[aug_indices]
                handle["point_source"][start:end] = point_source[aug_indices]
                handle["background_name"][start:end] = np.repeat(scene.name, aug_n)
                handle["background_library"][start:end] = np.repeat(scene.library, aug_n)
                handle["background_category"][start:end] = np.repeat(scene.category, aug_n)
                handle["alpha_multiplier"][start:end] = aug_alpha
                handle["augmented_flag"][start:end] = 1
                handle["source_row_type"][start:end] = aug_types

            cursor = next_cursor


def write_manifest(path: Path, split_map: Dict[str, List[BackgroundScene]]) -> None:
    lines = ["background_name,split,library,category,path"]
    for split_name, scenes in split_map.items():
        for scene in scenes:
            lines.append(f"{scene.name},{split_name},{scene.library},{scene.category},{scene.path.as_posix()}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    out_dir: Path,
    split_map: Dict[str, List[BackgroundScene]],
    dataset_path: Path,
    source_manifest: Path,
    train_plan: Dict[str, np.ndarray],
    alpha_plan: Dict[str, List[AugmentedRow]],
    mid_frac: float,
    high_frac: float,
    mid_range: Tuple[float, float],
    high_range: Tuple[float, float],
    alpha_min: float,
    alpha_max: float,
) -> None:
    total_mid = sum(sum(1 for row in rows if row.row_type == "alpha_mid") for rows in alpha_plan.values())
    total_high = sum(sum(1 for row in rows if row.row_type == "alpha_high") for rows in alpha_plan.values())
    lines = [
        f"source_dataset={dataset_path.as_posix()}",
        f"source_manifest={source_manifest.as_posix()}",
        "field_name=xco2_mixed_absolute",
        "split_policy=background_scene_level_same_as_make1",
        "train_sampling=category_balanced_on_make1_train_backgrounds_plus_alpha",
        f"train_target_per_category={TARGET_PER_CATEGORY_TRAIN}",
        f"train_alpha_mid_frac={mid_frac}",
        f"train_alpha_high_frac={high_frac}",
        f"train_alpha_mid_range={mid_range[0]}-{mid_range[1]}",
        f"train_alpha_high_range={high_range[0]}-{high_range[1]}",
        f"alpha_bounds={alpha_min}-{alpha_max}",
        f"train_alpha_mid_total={total_mid}",
        f"train_alpha_high_total={total_high}",
        "validation_alpha=disabled",
        "",
    ]
    for split_name, scenes in split_map.items():
        lines.append(f"[{split_name}] backgrounds={len(scenes)}")
        for scene in scenes:
            if split_name == "train":
                rows = alpha_plan.get(scene.name, [])
                n_mid = sum(1 for row in rows if row.row_type == "alpha_mid")
                n_high = sum(1 for row in rows if row.row_type == "alpha_high")
                lines.append(
                    f"- {scene.name} ({scene.library}, {scene.category}) "
                    f"base={len(train_plan[scene.name])} alpha_mid={n_mid} alpha_high={n_high}"
                )
            else:
                lines.append(f"- {scene.name} ({scene.library}, {scene.category})")
        lines.append("")
    (out_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build absolute balanced_v2 plus_alpha: fixed make1 validation, balanced train, alpha on train only."
    )
    parser.add_argument("--a-dir", required=True)
    parser.add_argument("--b-dir", required=True)
    parser.add_argument("--plume-dataset", required=True)
    parser.add_argument("--source-manifest", required=True, help="Path to make1 background_manifest.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mid-frac", type=float, default=DEFAULT_MID_FRAC)
    parser.add_argument("--high-frac", type=float, default=DEFAULT_HIGH_FRAC)
    parser.add_argument("--mid-low", type=float, default=20.0)
    parser.add_argument("--mid-high", type=float, default=30.0)
    parser.add_argument("--high-low", type=float, default=30.0)
    parser.add_argument("--high-high", type=float, default=40.0)
    parser.add_argument("--alpha-min", type=float, default=1.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    args = parser.parse_args()

    scene_lookup = list_background_scenes(Path(args.a_dir), Path(args.b_dir))
    split_map = load_split_map_from_manifest(Path(args.source_manifest), scene_lookup)
    plume_lib = load_plume_library(Path(args.plume_dataset))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_plan = build_balanced_train_plan(split_map["train"], plume_lib, seed=args.seed)
    alpha_plan = build_alpha_plan(
        split_map["train"],
        train_plan,
        plume_lib,
        seed=args.seed + 1000,
        mid_frac=args.mid_frac,
        high_frac=args.high_frac,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        mid_range=(args.mid_low, args.mid_high),
        high_range=(args.high_low, args.high_high),
    )
    n_plume_samples = plume_lib["plume"].shape[0]

    metadata = {
        "source_dataset": Path(args.plume_dataset).as_posix(),
        "source_manifest": Path(args.source_manifest).as_posix(),
        "field_name": "xco2_mixed_absolute",
        "seed": int(args.seed),
        "split_policy": "background_scene_level_same_as_make1",
        "train_sampling": "category_balanced_on_make1_train_backgrounds_plus_alpha",
        "n_backgrounds_total": sum(len(v) for v in split_map.values()),
        "n_point_sources": len(np.unique(plume_lib["point_source"])),
        "train_target_per_category": TARGET_PER_CATEGORY_TRAIN,
        "train_alpha_mid_frac": float(args.mid_frac),
        "train_alpha_high_frac": float(args.high_frac),
        "train_alpha_mid_range": f"{args.mid_low}-{args.mid_high}",
        "train_alpha_high_range": f"{args.high_low}-{args.high_high}",
        "alpha_min": float(args.alpha_min),
        "alpha_max": float(args.alpha_max),
        "validation_alpha": "disabled",
    }

    write_split(
        out_dir / "train.h5",
        split_map["train"],
        plume_lib,
        dict(metadata, split_name="train"),
        train_plan,
        alpha_plan=alpha_plan,
    )

    for split_name in ["valid", "valid_2", "extra_valid"]:
        write_split(
            out_dir / f"{split_name}.h5",
            split_map[split_name],
            plume_lib,
            dict(metadata, split_name=split_name),
            full_plan(split_map[split_name], n_plume_samples),
            alpha_plan=None,
        )

    write_manifest(out_dir / "background_manifest.csv", split_map)
    write_readme(
        out_dir,
        split_map,
        Path(args.plume_dataset),
        Path(args.source_manifest),
        train_plan,
        alpha_plan,
        args.mid_frac,
        args.high_frac,
        (args.mid_low, args.mid_high),
        (args.high_low, args.high_high),
        args.alpha_min,
        args.alpha_max,
    )


if __name__ == "__main__":
    main()
