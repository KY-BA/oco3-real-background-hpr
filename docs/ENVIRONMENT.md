# Environment

The formal training execution records identify a Python environment inspected during this release preparation. Its current Python is 3.8.20, TensorFlow GPU and Keras package metadata are 2.9.0, and the actual NumPy import reports 1.24.4. Details are in `manifests/environment_observed.json`.

`requirements.txt` reconstructs the directly used dependencies from this evidence and source imports. It is **not a complete historical lock file**, and installation into a newly created environment was not tested during packaging. The distributed TensorFlow package is listed as `tensorflow==2.9.0`; the inspected Windows environment records the `tensorflow-gpu` distribution. Keep TensorFlow and Keras on the compatible 2.9 line for the legacy H5 models rather than silently upgrading to Keras 3.

For example, after installing a suitable Python 3.8 environment:

```bash
python -m pip install -r requirements.txt
```

The standard-library result checker requires no TensorFlow. Inference and training require the packages above. Weights & Biases is optional and disabled in the manuscript configurations; no account or API token is needed.

Mixed-data loading materializes the training arrays and intermediate copies. The `195465 x 32 x 32 x 3` float32 input alone occupies about 2.24 GiB, so full training needs substantially more memory. Saved-model inference uses the separately exported training medians and does not load that array.

The numerical checks actually performed for this package, including the runtime used for inference, are recorded in `VALIDATION.md`. Training reproducibility can also depend on TensorFlow kernels, GPU libraries, and hardware; the release preserves seeds and configurations without claiming bitwise-identical retraining on every platform.
