# Upstream and third-party notices

## Upstream OCO-3 inversion implementation

Author: Joffrey Dumont Le Brazidec. Software archive: v1.0.2, released 2024-10-30, DOI [10.5281/zenodo.14013176](https://doi.org/10.5281/zenodo.14013176). Associated article: *Quantification of CO2 hotspot emissions from OCO-3 SAM CO2 satellite images using deep learning methods*, GMD 18, 3607-3622 (2025), DOI [10.5194/gmd-18-3607-2025](https://doi.org/10.5194/gmd-18-3607-2025).

The versioned Zenodo metadata identifies CC BY 4.0. The verified archive corresponds to commit `bdf7f24ff61e31283807891bfdbd2b534d7361e8`; its SHA-256 is `e165daefe6a6d150023de236f52190dd6caa68c46edf22f53b39392fc6d4e8bb`. The license text is included in `LICENSES/CC-BY-4.0.txt`. This evidence concerns this particular archived snapshot, not the current state or license of a later repository.

The supplied research project modifies several upstream files, including the data loader, training manager, callbacks/generators, model builders, and preprocessing. Existing author headers are preserved. `manifests/upstream.json` records original-versus-local hashes; `manifests/source_provenance.json` records the files selected for this release. Differences are not all asserted to be wholly original inventions of the present authors. The original README is retained separately as `docs/README_UPSTREAM.md`; its funding, contact, and official-repository statements refer to the upstream project.

## Component notices

- `models/my_efficientnet.py` retains the TensorFlow Authors, Pavel Yakubovskiy, and Björn Barz attribution and Apache 2.0 header. The Apache 2.0 license text is in `LICENSES/Apache-2.0.txt`.
- `models/Unet_backboned.py` identifies the segmentation_models source. The upstream repository's MIT license for that component, copyright 2018 Pavel Yakubovskiy, is retained in `LICENSES/segmentation-models-MIT.txt`. The complete vendored segmentation_models tree is not needed for the selected imports and is not duplicated here.
- The unused `models/my_squeezenet.py` implementation was excluded from this manuscript release because its separately identified Christopher Masch source did not provide an explicit license. Its import was removed and the SqueezeNet-only model-selector branch disabled. The CNN/HPR branches are unchanged; see `docs/release_source_changes.diff`.

Python packages installed through `requirements.txt` retain their own licenses. Satellite and simulation products retain their respective providers' terms. These notices do not relicense third-party materials.

## Original project contributions

The authors' original code contributions are licensed under MIT, as recorded in `LICENSE_SCOPE.md` and `LICENSES/MIT.txt`. This grant covers their original portions of modified files and their original new code, including the release scripts. It does not replace any upstream or third-party license, attribution, or copyright notice listed above.
