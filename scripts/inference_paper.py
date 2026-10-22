"""Run the paper's saved models on the explicitly listed 33 OCO-3 scenes.

Use the optional assets archive alongside the code release. This script performs
inference only. It does not fit normalization, estimate mixed-data fill values
from observations, select new scenes, or train models.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True,
                        help="Directory containing checkpoints/ and observations/.")
    parser.add_argument("--model", choices=["all", "pure_cnn", "mixed_cnn", "hpr"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Default: all five seeds in the paper protocol.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/evaluation_scenes.csv")
    parser.add_argument("--protocol", type=Path, default=ROOT / "manifests/paper_protocol.json")
    parser.add_argument("--fill-values", type=Path, default=ROOT / "manifests/mixed_fill_values.json")
    return parser.parse_args()


def main():
    args = parse_arguments()
    protocol = read_json(args.protocol)
    expected_seeds = [int(seed) for seed in protocol["seeds"]]
    if expected_seeds != [42, 52, 62, 72, 82]:
        raise ValueError("The protocol does not contain the paper's five seeds.")
    seeds = expected_seeds if args.seeds is None else args.seeds
    if not seeds or len(seeds) != len(set(seeds)) or any(seed not in expected_seeds for seed in seeds):
        raise ValueError("Seeds must be unique members of 42, 52, 62, 72, 82.")
    expected_models = ["pure_cnn", "mixed_cnn", "hpr"]
    if protocol["models"] != expected_models:
        raise ValueError("Unexpected model list in the paper protocol.")
    models = expected_models if args.model == "all" else [args.model]
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        scenes = list(csv.DictReader(handle))
    if len(scenes) != 33:
        raise ValueError("Expected the fixed 33-scene manifest; no automatic scene selection is performed.")
    references = protocol["annual_emission_references_mt_co2_yr"]
    if len(references) != 8 or set(row["plant"] for row in scenes) != set(references):
        raise ValueError("Scene plants differ from the eight-plant protocol.")
    keys = [(row["plant"], row["scene"]) for row in scenes]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate plant/scene rows in the fixed manifest.")
    for row in scenes:
        if Path(row["scene"]).name != row["scene"] or "\ufeff" in row["scene"]:
            raise ValueError("Scene names must be clean basenames: " + repr(row["scene"]))
        row["plant_lat"] = float(row["plant_lat"])
        row["plant_lon"] = float(row["plant_lon"])
        row["path"] = args.assets_dir / "observations" / row["plant"] / row["scene"]
        if not row["path"].is_file():
            raise FileNotFoundError(str(row["path"]))
    for model_name in models:
        for seed in seeds:
            model_dir = args.assets_dir / "checkpoints" / model_name / ("seed_%d" % seed)
            for filename in ["w_best.h5", "norm_layer.keras", "config.yaml"]:
                if not (model_dir / filename).is_file():
                    raise FileNotFoundError(str(model_dir / filename))
    output_csv = args.output_dir / "predictions.csv"
    output_meta = args.output_dir / "inference_metadata.json"
    if output_csv.exists() or output_meta.exists():
        raise FileExistsError("Choose a new output directory; existing prediction files are not overwritten.")

    # Delay scientific imports so --help and input preflight need no TensorFlow.
    import numpy as np
    import tensorflow as tf
    from tools.compare_puresim_vs_mixed_on_real_oco3 import (
        apply_norm, fill_per_channel_median, grid_three_channels,
        load_cfg_fields, load_model as load_pure_model,
        load_norm_spec, load_raw_points,
    )
    from tools.infer_real_oco3_absolute import (
        load_model as load_mixed_model, load_norm_model,
        prepare_input_from_nc4, select_emission_output,
    )

    fill = None
    fill_values = None
    if any(name != "pure_cnn" for name in models):
        fill = read_json(args.fill_values)
        if fill["field_name"] != "xco2_mixed_absolute_uv" or fill["channels"] != ["XCO2", "u", "v"]:
            raise ValueError("Mixed-training fill metadata has unexpected channels or field.")
        fill_values = np.asarray(fill["fill_values"], dtype=np.float32)
        if fill_values.shape != (3,) or not np.isfinite(fill_values).all():
            raise ValueError("Expected three finite float32 training-subset medians.")

    predictions = []
    checkpoint_hashes = []
    for model_name in models:
        for seed in seeds:
            model_dir = args.assets_dir / "checkpoints" / model_name / ("seed_%d" % seed)
            print("Inference: %s, seed %d" % (model_name, seed), flush=True)
            if model_name == "pure_cnn":
                config_fields = load_cfg_fields(model_dir)
                norm = load_norm_spec(model_dir)
                model = load_pure_model(model_dir, "w_best.h5")
            else:
                norm = load_norm_model(model_dir)
                # Original helper rebuilds HPR from config before loading its
                # weights; the mixed CNN uses the saved full Keras model.
                model = load_mixed_model(model_dir, "w_best.h5")
            checkpoint_hashes.append({
                "model": model_name, "seed": seed,
                "w_best_sha256": sha256(model_dir / "w_best.h5"),
                "norm_layer_sha256": sha256(model_dir / "norm_layer.keras"),
                "config_sha256": sha256(model_dir / "config.yaml"),
            })
            for row in scenes:
                if model_name == "pure_cnn":
                    points = load_raw_points(row["path"], quality_filter=True)
                    xco2, u, v, valid_mask = grid_three_channels(
                        points, lat0=row["plant_lat"], lon0=row["plant_lon"])
                    channels = {"xco2": xco2, "u_wind": u, "u_wind_train": u,
                                "v_wind": v, "v_wind_train": v}
                    selected = []
                    for key in ["chan_0", "chan_1", "chan_2", "chan_3", "chan_4"]:
                        name = str(config_fields.get(key, "None"))
                        if name == "None":
                            continue
                        if name not in channels:
                            raise ValueError("Unsupported pure-simulation input channel: " + name)
                        selected.append(channels[name])
                    if not selected:
                        raise ValueError("Pure-simulation configuration contains no input channels.")
                    raw = fill_per_channel_median(np.stack(selected, axis=-1))
                    x_norm = apply_norm(raw[np.newaxis, ...], norm)
                    valid_pixels = int(valid_mask.sum())
                else:
                    prepared = prepare_input_from_nc4(
                        row["path"], field_name=fill["field_name"],
                        fill_values=fill_values, norm_model=norm,
                        plant_lat=row["plant_lat"], plant_lon=row["plant_lon"])
                    x_norm = prepared["x_norm"]
                    valid_pixels = int(prepared["valid_mask"].sum())
                prediction = float(np.squeeze(select_emission_output(model.predict(x_norm, verbose=0))))
                if not np.isfinite(prediction):
                    raise ValueError("Nonfinite prediction for %s/%s" % (row["plant"], row["scene"]))
                predictions.append({"model": model_name, "seed": seed,
                                    "plant": row["plant"], "scene": row["scene"],
                                    "predicted_emission": prediction, "valid_pixels": valid_pixels})
            del model, norm
            tf.keras.backend.clear_session()

    expected_rows = len(models) * len(seeds) * len(scenes)
    if len(predictions) != expected_rows:
        raise RuntimeError("Prediction row count differs from the requested fixed protocol.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", "plant", "scene", "predicted_emission", "valid_pixels"])
        writer.writeheader()
        writer.writerows(predictions)
    metadata = {
        "schema_version": "1.0", "operation": "inference_only",
        "models": models, "seeds": seeds, "n_scenes": 33, "n_plants": 8,
        "prediction_rows": len(predictions), "emission_unit": "Mt CO2 yr-1",
        "manifest_sha256": sha256(args.manifest), "protocol_sha256": sha256(args.protocol),
        "mixed_fill_values_sha256": sha256(args.fill_values) if fill is not None else None,
        "checkpoint_hashes": checkpoint_hashes,
        "tensorflow_version": tf.__version__, "numpy_version": np.__version__,
        "note": "Scene predictions only. Annual emission references are not instantaneous scene ground truth. Ensemble and individual-seed metrics must be aggregated separately.",
    }
    with output_meta.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print("Saved %d predictions to %s" % (len(predictions), output_csv))


if __name__ == "__main__":
    main()
