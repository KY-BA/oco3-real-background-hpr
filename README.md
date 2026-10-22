# OCO-3 real-background mixed training and HPR

Code and reproduction materials for **Real-Background Mixed Training and Plume-Aware Representation for OCO-3 Power-Plant CO2 Emission Estimation**, by Jiahao Chen, Ye Deng, and Guixue Cheng.

This release separates the manuscript configurations from exploratory experiments in the original working directory. It contains the original training/model implementations, portable launchers, fixed data manifests, 13 gridded backgrounds, and the saved predictions underlying the reported metrics. No new training results are introduced by preparing this release.

## Start with the saved results

From this directory, using Python 3.8 or later:

```bash
python scripts/verify_results.py
```

This command needs only the Python standard library. It validates all 495 predictions and recalculates ensemble, individual-seed, plant, emission-group, and appendix metrics. It writes its outputs to `runs/reproduced_metrics/`; it does not run inference or training.

| Model | Ensemble MAE (Mt CO2 yr-1) | Ensemble MAPE (%) |
|---|---:|---:|
| Pure-simulation CNN | 10.14094549 | 49.54281013 |
| Real-background mixed CNN | 5.79925178 | 28.28333730 |
| HPR | 4.72256693 | 20.75667053 |

The evaluation uses **eight plants, 33 scenes, and seeds 42, 52, 62, 72, and 82**. Predictions are averaged over seeds within each scene, then over scenes within each plant. Errors are finally averaged equally over the eight plants. The mean of individual-seed errors is a different statistic. Annual emission references are not instantaneous ground truth at the satellite overpass.

The appendix uses a **post hoc plant-level** rule on this same benchmark: select HPR when its plant prediction is below 10 or above 16 Mt CO2 yr-1, and the mixed CNN otherwise. Its MAE is 4.48407695 and MAPE is 19.11524692%. This is not a trained gate or an independent validation.

## Run the saved models

Install the direct dependencies in a separate environment (see `docs/ENVIRONMENT.md`). Obtain the accompanying reproduction-assets archive and extract it alongside this repository:

```text
parent/
  oco3-real-background-hpr/
  reproduction-assets/
    checkpoints/{pure_cnn,mixed_cnn,hpr}/seed_{42,52,62,72,82}/
    observations/{plant}/{scene}.nc4
```

Then run:

```bash
python scripts/verify_package.py --assets-dir ../reproduction-assets
python scripts/inference_paper.py --assets-dir ../reproduction-assets --output-dir runs/fresh_inference
python scripts/verify_results.py --predictions runs/fresh_inference/predictions.csv --report-only --output-dir runs/fresh_metrics
```

The inference launcher reads only the fixed scene manifest; it never adds other `.nc4` files by scanning directories. It loads the published normalization models and the training-subset medians in `manifests/mixed_fill_values.json`. It does not estimate mixed-training preprocessing statistics from the evaluation scenes. The large mixed training H5 is therefore not needed for inference. `--report-only` records comparisons without claiming exact equality. The CPU rerun during packaging reproduced the main metrics at the paper's reporting precision but showed small HPR differences from the archived predictions; the actual differences are retained in `docs/VALIDATION.md`. The saved-result check in the first section remains strict.

To test one model/seed first, add `--model hpr --seeds 42` and use a different output directory. The full metric checker requires all three models and all five seeds.

## Train from the manuscript configurations

The CNN/HPR definitions and training logic are unchanged from the supplied research project. One unused SqueezeNet import was removed and its non-paper branch disabled so that this unrelated implementation need not be redistributed; the exact change is in `docs/release_source_changes.diff`. `configs/training/` contains three complete, verified seed-42 configurations; only data/output locations and the run label were made portable. The launcher substitutes the requested paper seeds. All three models use Adam, learning rate 0.0005, dropout 0.20, batch size 32, and a maximum of 500 epochs, with the archived validation callbacks.

```bash
python scripts/train_paper.py --model hpr --data-dir /path/to/mixed_xco2uv --output-dir /path/to/new_training_runs
python scripts/train_paper.py --model mixed_cnn --data-dir /path/to/mixed_xco2uv --output-dir /path/to/new_training_runs
python scripts/train_paper.py --model pure_cnn --data-dir /path/to/smartcarb_parent --output-dir /path/to/new_training_runs
```

These commands perform preflight checks and print the plan. **Add `--execute` to start training.** Existing run directories are rejected. For paths containing spaces, enclose the path in quotes. Mixed training requires `train.h5`, `valid.h5`, `valid_2.h5`, and `extra_valid.h5`. Pure-simulation training expects `2km_forOco3_32/dataset.nc` below the specified parent; its archived loader uses the dataset's internal partition information.

Use these launchers rather than the original `main.py` defaults in `cfg/`, which retain the historical project layout. The large training datasets are not included in the code archive. See `docs/DATA_AND_REPRODUCTION.md` for construction steps and source records.

## Contents and provenance

| Path | Purpose |
|---|---|
| `main.py`, `Data.py`, `model_training.py`, `include/`, `models/`, `data/utils.py` | Original training, preprocessing, and model implementations |
| `configs/training/` | Three manuscript configurations |
| `scripts/` | New release launchers, metric checker, and file-integrity checker |
| `tools/` | Selected original data-building and inference helpers |
| `data/backgrounds/` | Thirteen selected background NPZs and diagnostic metadata |
| `manifests/` | Scene/background splits, checkpoint and input hashes, protocol, source attribution |
| `results/reference/` | Saved predictions and independently recalculated reference metrics |
| `docs/` | Environment, data construction, provenance, and reproduction scope |
| `LICENSES/`, `THIRD_PARTY_NOTICES.md` | Upstream and component license information |

The original helpers include legacy table/plot exporters. Their automatically generated prose is not the current manuscript, and they are not recommended publication-text generators. The release launchers call the verified numerical routines directly. Unused architecture definitions remain where needed by the original static imports; their presence does not mean they were evaluated in this manuscript.

This project builds on Joffrey Dumont Le Brazidec's [v1.0.2 software archive](https://doi.org/10.5281/zenodo.14013176), commit `bdf7f24ff61e31283807891bfdbd2b534d7361e8`, accompanying the [GMD 2025 paper](https://doi.org/10.5194/gmd-18-3607-2025). The versioned archive records CC BY 4.0. Original author headers and applicable third-party notices are retained. The authors' original code contributions, including the release scripts, are available under the MIT License; upstream and third-party code retain their respective terms. See `LICENSE`, `LICENSE_SCOPE.md`, and `LICENSES/` for the scope and full texts. This is a manuscript preparation package; no GitHub URL, manuscript DOI, or archive DOI for this new release has been assigned.

Actual checks and their limits are documented in `docs/VALIDATION.md`. In particular, preparing this package did not rerun full model training or the entire multi-gigabyte dataset construction.
