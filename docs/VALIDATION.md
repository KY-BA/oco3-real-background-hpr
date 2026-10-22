# Validation performed during packaging

Date: 22 October 2026. These checks concern this preparation package. They do not claim a new training experiment or a clean-environment installation test.

## Source, configurations, and inputs

- The retained 27 core training/data/evaluation source files match the supplied project byte for byte. `models/reg.py` has only the disclosed removal of the unused SqueezeNet import and branch; the CNN/HPR branches are unchanged. See `release_source_changes.diff`.
- Each of the three complete manuscript configurations was compared recursively with its formal seed-42 record. Only `data.path.directory`, `dir_res`, and `exp_name` differ. The launcher additionally selects the requested seed and Hydra run directory; scientific hyperparameters are preserved.
- Python 3.8 syntax and local imports were checked. Actual `main.py --cfg job --resolve` calls succeeded for all three configurations without beginning training. The training launcher passed dry-run, missing-input, existing-output, and paths-with-spaces checks.
- Original data-builder interfaces ran with `--help`. The first builder successfully read the 13 released backgrounds and their manifest, yielding 9/2/1/1 training/validation partitions. Complete dataset construction was not rerun.
- The 15 checkpoints and normalization files, 33 observation files, and 13 background arrays have recorded SHA-256 values. Exact mixed-training channel medians were calculated from the training subset, with its input hash retained.

The installed optional `wandb` package raised Windows permission warnings while cleaning temporary directories at process exit during configuration checks. All three configuration processes returned 0 after printing their resolved configurations. `wandb` is disabled in the paper configurations and is not required by the release dependency list. This environment warning was not concealed or fixed by modifying the research code.

## Reaggregation of archived predictions: PASS

`scripts/verify_results.py` checked all 495 distinct model/seed/scene records, identical scene membership across models/seeds, eight plant references, seed aggregation, plant aggregation, individual-seed summaries, and the appendix's post hoc plant-level rule. All 12 checked ensemble/appendix aggregate values match the independently calculated reference values at absolute tolerance 1e-9. See `validation/metrics/verification.json` and the accompanying tables.

This arithmetic check uses the archived predictions. It is distinct from rerunning the model.

## Fresh model inference: completed, numerical differences retained

All 495 predictions were regenerated using the released checkpoints and observational inputs in the existing Python 3.8.20 / TensorFlow 2.9.0 / Keras 2.9.0 / NumPy 1.24.4 environment. The full run used CPU execution (`CUDA_VISIBLE_DEVICES=-1`) and one TF intra-op, TF inter-op, and OMP thread. The scientific files used in this run were unchanged except that the unused SqueezeNet import/branch was subsequently removed; the final CNN/HPR imports were checked after that change.

| Model | Archived MAE | Fresh CPU MAE | Archived MAPE (%) | Fresh CPU MAPE (%) | Largest scene/seed absolute difference |
|---|---:|---:|---:|---:|---:|
| Pure-simulation CNN | 10.140945488 | 10.140945591 | 49.542810131 | 49.542811037 | 0.000004768 |
| Mixed CNN | 5.799251775 | 5.799251246 | 28.283337301 | 28.283333007 | 0.000009060 |
| HPR | 4.722566926 | 4.722860654 | 20.756670528 | 20.758772112 | 0.022829056 |

MAE and prediction differences are in Mt CO2 yr-1. Main ensemble metrics agree at the manuscript's reporting precision. Fresh inference does **not** satisfy the strict archived-result equality check; `validation/fresh_metrics/verification.json` is deliberately marked `DIFFERENT`. The archived predictions were not replaced and the default tolerance was not relaxed.

For HPR, mean absolute scene/seed difference is 0.000893532 and 130 of 165 differences exceed 1e-4. Checkpoint pairing, scene membership, valid-pixel counts, and the training-median source showed no mismatch. Independent CPU and GPU trials for the largest-difference example demonstrated backend/kernel sensitivity; the GPU runs did not recover the archived value either. These trials do not establish that backend differences explain the entire discrepancy. The full cause remains unconfirmed.

Use the supplied launcher to rebuild HPR and load its weights. Directly deserializing the Lambda-containing H5 graph failed with `unknown opcode` in a diagnostic trial. The original helper already uses the reconstruction route; its model mathematics were not rewritten for this package.

Evidence is kept separately from `results/reference/`: `validation/fresh_inference/`, `validation/fresh_metrics/`, `validation/prediction_comparison.json`, `validation/hpr_difference_summary.json`, and `validation/hpr_device_diagnostic.json`.

## Release integrity and remaining coverage

`scripts/verify_package.py` validates file sizes and SHA-256 values for the code package and, when requested, the separate assets. `.gitattributes` disables automatic line-ending conversion so the release hashes remain meaningful. The hash manifest excludes itself. Editing the package after preparation requires an updated integrity manifest; this checker is for the prepared snapshot, not arbitrary future commits.

No full model retraining, full mixed-data rebuild, or installation into a clean environment was performed. Exact final intermediate arrays for Figure 8 were not identified; older diagnostic arrays from other checkpoints were not substituted. Other reproduction coverage is described in `DATA_AND_REPRODUCTION.md`.
