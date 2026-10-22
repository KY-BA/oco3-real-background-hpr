# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import logging
import sys
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, List, Optional, Union

import h5py
import numpy as np
import tensorflow as tf
from icecream import ic

import data.utils as data_utils
from models.preprocessing import (
    CloudsLayer,
    ConditionalNoiseLayer,
    TrainingTimeNormalization,
)


def make_emission_bins(
    y: np.ndarray,
    low_threshold: float = 8.0,
    mid_threshold: float = 18.0,
    high_threshold: float = 30.0,
) -> np.ndarray:
    """Map emission values to low/mid/high/very-high bin ids."""
    values = np.asarray(y).reshape(-1)
    bins = np.zeros(values.shape[0], dtype=np.int32)
    bins[values >= low_threshold] = 1
    bins[values >= mid_threshold] = 2
    bins[values >= high_threshold] = 3
    return bins


@dataclass
class PrecomputedDataGen(tf.keras.utils.Sequence):
    """Simple generator for precomputed mixed-background training arrays."""

    x: Any
    y: np.ndarray
    batch_size: int = 128
    shuffle: bool = True
    norm_noise_std_min: float = 0.0
    norm_noise_std_max: float = 0.0

    def __post_init__(self):
        if isinstance(self.x, np.ndarray):
            self.N_data = self.x.shape[0]
        elif isinstance(self.x, list) and all(
            isinstance(arr, np.ndarray) for arr in self.x
        ):
            self.N_data = self.x[0].shape[0]
        else:
            ic("Unsupported input type for PrecomputedDataGen.")
            sys.exit()
        self.indices = np.arange(self.N_data)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index: int):
        batch_ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        if isinstance(self.x, np.ndarray):
            x_batch = self.x[batch_ids]
        else:
            x_batch = [arr[batch_ids] for arr in self.x]
        x_batch = self._apply_norm_noise(x_batch)
        y_batch = self.y[batch_ids]
        return x_batch, y_batch

    def __len__(self):
        return int(np.ceil(self.N_data / self.batch_size))

    def _apply_norm_noise(self, x_batch):
        if self.norm_noise_std_max <= 0:
            return x_batch
        sigma = np.random.uniform(self.norm_noise_std_min, self.norm_noise_std_max)
        if isinstance(x_batch, np.ndarray):
            return (
                x_batch
                + np.random.normal(0.0, sigma, size=x_batch.shape).astype(np.float32)
            ).astype(np.float32)
        out = list(x_batch)
        out[0] = (
            out[0]
            + np.random.normal(0.0, sigma, size=out[0].shape).astype(np.float32)
        ).astype(np.float32)
        return out


@dataclass
class PrecomputedSetDataGen(tf.keras.utils.Sequence):
    """Generator that groups same-emission mixed scenes into K-scene sets."""

    x: Any
    y: np.ndarray
    batch_size: int = 128
    shuffle: bool = True
    set_size: int = 4
    group_decimals: int = 5
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.x, np.ndarray):
            ic("PrecomputedSetDataGen currently expects one ndarray input.")
            sys.exit()
        self.N_data = self.x.shape[0]
        self.y_flat = np.asarray(self.y).reshape(-1).astype(np.float32)
        self.indices = np.arange(self.N_data)
        self.rng = np.random.default_rng(self.seed)
        self.groups = self._build_groups(self.y_flat)
        self.set_indices = self._make_set_indices()
        self.on_epoch_end()

    def _build_groups(self, y_values: np.ndarray) -> dict:
        groups = defaultdict(list)
        keys = np.round(y_values, self.group_decimals)
        for idx, key in enumerate(keys):
            groups[float(key)].append(idx)
        return {
            key: np.asarray(value, dtype=np.int64)
            for key, value in groups.items()
        }

    def _make_set_indices(self) -> np.ndarray:
        set_indices = np.empty((self.N_data, self.set_size), dtype=np.int64)
        keys = np.round(self.y_flat, self.group_decimals)
        for idx, key in enumerate(keys):
            members = self.groups[float(key)]
            replace = members.shape[0] < self.set_size
            chosen = self.rng.choice(members, size=self.set_size, replace=replace)
            if idx not in chosen:
                chosen[0] = idx
            set_indices[idx] = chosen
        return set_indices

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            self.set_indices = self._make_set_indices()

    def __getitem__(self, index: int):
        batch_ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        x_batch = self.x[self.set_indices[batch_ids]]
        y_batch = self.y_flat[batch_ids]
        return x_batch, y_batch

    def __len__(self):
        return int(np.ceil(self.N_data / self.batch_size))


@dataclass
class PrecomputedWeightedDataGen(tf.keras.utils.Sequence):
    """Precomputed generator with emission-dependent sample weights."""

    x: Any
    y: np.ndarray
    batch_size: int = 128
    shuffle: bool = True
    high_threshold: float = 18.0
    very_high_threshold: float = 30.0
    high_weight: float = 1.5
    very_high_weight: float = 2.0

    def __post_init__(self):
        if isinstance(self.x, np.ndarray):
            self.N_data = self.x.shape[0]
        elif isinstance(self.x, list) and all(
            isinstance(arr, np.ndarray) for arr in self.x
        ):
            self.N_data = self.x[0].shape[0]
        else:
            ic("Unsupported input type for PrecomputedWeightedDataGen.")
            sys.exit()
        self.indices = np.arange(self.N_data)
        self.sample_weights = self._make_sample_weights(self.y)
        self.on_epoch_end()

    def _make_sample_weights(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y).reshape(-1)
        weights = np.ones(values.shape[0], dtype=np.float32)
        weights[values >= self.high_threshold] = np.float32(self.high_weight)
        weights[values >= self.very_high_threshold] = np.float32(
            self.very_high_weight
        )
        return weights

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index: int):
        batch_ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        if isinstance(self.x, np.ndarray):
            x_batch = self.x[batch_ids]
        else:
            x_batch = [arr[batch_ids] for arr in self.x]
        y_batch = self.y[batch_ids]
        return x_batch, y_batch, self.sample_weights[batch_ids]

    def __len__(self):
        return int(np.ceil(self.N_data / self.batch_size))


@dataclass
class PrecomputedBinBalancedDataGen(tf.keras.utils.Sequence):
    """Precomputed generator that samples emission bins uniformly per epoch."""

    x: Any
    y: np.ndarray
    batch_size: int = 128
    shuffle: bool = True
    low_threshold: float = 8.0
    mid_threshold: float = 18.0
    high_threshold: float = 30.0
    norm_noise_std_min: float = 0.0
    norm_noise_std_max: float = 0.0

    def __post_init__(self):
        if isinstance(self.x, np.ndarray):
            self.N_data = self.x.shape[0]
        elif isinstance(self.x, list) and all(
            isinstance(arr, np.ndarray) for arr in self.x
        ):
            self.N_data = self.x[0].shape[0]
        else:
            ic("Unsupported input type for PrecomputedBinBalancedDataGen.")
            sys.exit()
        self.bin_labels = make_emission_bins(
            self.y,
            low_threshold=self.low_threshold,
            mid_threshold=self.mid_threshold,
            high_threshold=self.high_threshold,
        )
        self.bin_indices = [
            np.where(self.bin_labels == bin_id)[0] for bin_id in range(4)
        ]
        self.bin_indices = [idx for idx in self.bin_indices if idx.shape[0] > 0]
        self.indices = np.arange(self.N_data)
        self.on_epoch_end()

    def on_epoch_end(self):
        n_bins = len(self.bin_indices)
        n_per_bin = int(np.ceil(self.N_data / n_bins))
        sampled = [
            np.random.choice(idx, size=n_per_bin, replace=idx.shape[0] < n_per_bin)
            for idx in self.bin_indices
        ]
        self.indices = np.concatenate(sampled)[: self.N_data]
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index: int):
        batch_ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        if isinstance(self.x, np.ndarray):
            x_batch = self.x[batch_ids]
        else:
            x_batch = [arr[batch_ids] for arr in self.x]
        x_batch = self._apply_norm_noise(x_batch)
        y_batch = self.y[batch_ids]
        return x_batch, y_batch

    def __len__(self):
        return int(np.ceil(self.N_data / self.batch_size))

    def _apply_norm_noise(self, x_batch):
        if self.norm_noise_std_max <= 0:
            return x_batch
        sigma = np.random.uniform(self.norm_noise_std_min, self.norm_noise_std_max)
        if isinstance(x_batch, np.ndarray):
            return (
                x_batch
                + np.random.normal(0.0, sigma, size=x_batch.shape).astype(np.float32)
            ).astype(np.float32)
        out = list(x_batch)
        out[0] = (
            out[0]
            + np.random.normal(0.0, sigma, size=out[0].shape).astype(np.float32)
        ).astype(np.float32)
        return out


@dataclass
class PrecomputedScaleAwareDataGen(tf.keras.utils.Sequence):
    """Generator for emission regression plus emission-bin classification."""

    x: Any
    y: np.ndarray
    batch_size: int = 128
    shuffle: bool = True
    bin_balanced: bool = False
    low_threshold: float = 8.0
    mid_threshold: float = 18.0
    high_threshold: float = 30.0
    n_bins: int = 4

    def __post_init__(self):
        if isinstance(self.x, np.ndarray):
            self.N_data = self.x.shape[0]
        elif isinstance(self.x, list) and all(
            isinstance(arr, np.ndarray) for arr in self.x
        ):
            self.N_data = self.x[0].shape[0]
        else:
            ic("Unsupported input type for PrecomputedScaleAwareDataGen.")
            sys.exit()
        self.bin_labels = make_emission_bins(
            self.y,
            low_threshold=self.low_threshold,
            mid_threshold=self.mid_threshold,
            high_threshold=self.high_threshold,
        )
        self.bin_indices = [
            np.where(self.bin_labels == bin_id)[0] for bin_id in range(4)
        ]
        self.bin_indices = [idx for idx in self.bin_indices if idx.shape[0] > 0]
        self.indices = np.arange(self.N_data)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.bin_balanced:
            n_bins = len(self.bin_indices)
            n_per_bin = int(np.ceil(self.N_data / n_bins))
            sampled = [
                np.random.choice(idx, size=n_per_bin, replace=idx.shape[0] < n_per_bin)
                for idx in self.bin_indices
            ]
            self.indices = np.concatenate(sampled)[: self.N_data]
            if self.shuffle:
                np.random.shuffle(self.indices)
        elif self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index: int):
        batch_ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        if isinstance(self.x, np.ndarray):
            x_batch = self.x[batch_ids]
        else:
            x_batch = [arr[batch_ids] for arr in self.x]
        y_batch = {
            "emission": self.y[batch_ids],
            "emission_bin": tf.keras.utils.to_categorical(
                self.bin_labels[batch_ids],
                num_classes=self.n_bins,
            ).astype(np.float32),
        }
        return x_batch, y_batch

    def __len__(self):
        return int(np.ceil(self.N_data / self.batch_size))


@dataclass
class PrecomputedCounterfactualDataGen(tf.keras.utils.Sequence):
    """Generator with adjacent counterfactual pairs for scale-consistent losses."""

    x: Any
    y: np.ndarray
    h5_path: str
    batch_size: int = 128
    shuffle: bool = True
    include_singletons: bool = True

    def __post_init__(self):
        if self.batch_size % 2 != 0:
            logging.error("PrecomputedCounterfactualDataGen requires an even batch_size.")
            sys.exit()
        if isinstance(self.x, np.ndarray):
            self.N_data = self.x.shape[0]
        elif isinstance(self.x, list) and all(
            isinstance(arr, np.ndarray) for arr in self.x
        ):
            self.N_data = self.x[0].shape[0]
        else:
            ic("Unsupported input type for PrecomputedCounterfactualDataGen.")
            sys.exit()

        self.groups = self._build_groups(self.h5_path)
        if len(self.groups) == 0:
            logging.error("No valid counterfactual/self-pair groups found.")
            sys.exit()
        self.group_order = np.arange(len(self.groups))
        self.pair_indices = None
        self.on_epoch_end()

    def _build_groups(self, h5_path: str) -> List[np.ndarray]:
        with h5py.File(h5_path, "r") as handle:
            source_idx = np.asarray(handle["source_idx_img"][:]).reshape(-1)
            alpha = np.asarray(handle["alpha_multiplier"][:]).reshape(-1)
            background = handle["background_name"][:]
        if source_idx.shape[0] != self.N_data:
            logging.error(
                "Counterfactual metadata length does not match loaded input length."
            )
            sys.exit()

        buckets = defaultdict(list)
        for idx, (src, a_value, bg_value) in enumerate(
            zip(source_idx, alpha, background)
        ):
            if isinstance(bg_value, (bytes, bytearray)):
                bg_key = bg_value.decode("utf-8", errors="replace")
            else:
                bg_key = str(bg_value)
            key = (int(src), float(np.round(a_value, 6)))
            buckets[key].append((idx, bg_key))

        groups = []
        for rows in buckets.values():
            if len(rows) == 1:
                if self.include_singletons:
                    groups.append(np.array([rows[0][0]], dtype=np.int64))
                continue
            by_background = defaultdict(list)
            for idx, bg_key in rows:
                by_background[bg_key].append(idx)
            bg_keys = list(by_background.keys())
            if len(bg_keys) >= 2:
                groups.append(
                    np.array(
                        [
                            np.random.choice(by_background[bg_key])
                            for bg_key in bg_keys
                        ],
                        dtype=np.int64,
                    )
                )
            elif self.include_singletons:
                groups.append(np.array([rows[0][0]], dtype=np.int64))
        return groups

    def _sample_pair(self, group: np.ndarray) -> np.ndarray:
        if group.shape[0] >= 2:
            return np.random.choice(group, size=2, replace=False).astype(np.int64)
        return np.array([group[0], group[0]], dtype=np.int64)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.group_order)
        pairs = [self._sample_pair(self.groups[idx]) for idx in self.group_order]
        self.pair_indices = np.asarray(pairs, dtype=np.int64)

    def __getitem__(self, index: int):
        start = index * (self.batch_size // 2)
        end = min(start + (self.batch_size // 2), self.pair_indices.shape[0])
        batch_ids = self.pair_indices[start:end].reshape(-1)
        if isinstance(self.x, np.ndarray):
            x_batch = self.x[batch_ids]
        else:
            x_batch = [arr[batch_ids] for arr in self.x]
        y_batch = self.y[batch_ids]
        return x_batch, y_batch

    def __len__(self):
        return int(np.ceil(len(self.groups) / (self.batch_size // 2)))


@dataclass
class PrecomputedConsistencyNoiseDataGen(tf.keras.utils.Sequence):
    """Generator returning clean/noisy pairs for perturbation-consistency losses."""

    x: Any
    y: np.ndarray
    batch_size: int = 128
    shuffle: bool = True
    norm_noise_std_min: float = 0.0
    norm_noise_std_max: float = 0.3
    low_threshold: float = 8.0

    def __post_init__(self):
        if isinstance(self.x, np.ndarray):
            self.N_data = self.x.shape[0]
        elif isinstance(self.x, list) and all(
            isinstance(arr, np.ndarray) for arr in self.x
        ):
            self.N_data = self.x[0].shape[0]
        else:
            ic("Unsupported input type for PrecomputedConsistencyNoiseDataGen.")
            sys.exit()
        self.indices = np.arange(self.N_data)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index: int):
        batch_ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        if isinstance(self.x, np.ndarray):
            clean = self.x[batch_ids].astype(np.float32)
            noisy = self._make_noisy(clean)
            x_batch = np.concatenate([clean, noisy], axis=0).astype(np.float32)
        else:
            clean0 = self.x[0][batch_ids].astype(np.float32)
            noisy0 = self._make_noisy(clean0)
            x_batch = [np.concatenate([clean0, noisy0], axis=0).astype(np.float32)]
            for arr in self.x[1:]:
                x_batch.append(np.concatenate([arr[batch_ids], arr[batch_ids]], axis=0))

        y_base = self.y[batch_ids].astype(np.float32)
        low_flag = (y_base.reshape(-1) <= self.low_threshold).astype(np.float32)[:, None]
        y_meta = np.concatenate([y_base, low_flag], axis=1)
        y_batch = np.concatenate([y_meta, y_meta], axis=0).astype(np.float32)
        return x_batch, y_batch

    def __len__(self):
        return int(np.ceil(self.N_data / self.batch_size))

    def _make_noisy(self, clean: np.ndarray) -> np.ndarray:
        sigma = np.random.uniform(self.norm_noise_std_min, self.norm_noise_std_max)
        if sigma <= 0:
            return np.array(clean, copy=True, dtype=np.float32)
        return (
            clean
            + np.random.normal(0.0, sigma, size=clean.shape).astype(np.float32)
        ).astype(np.float32)


@dataclass
class PrecomputedMultiTaskDataGen(tf.keras.utils.Sequence):
    """Generator for mixed-background emission regression plus plume-map auxiliary."""

    x: Any
    y_emission: np.ndarray
    y_plume: np.ndarray
    batch_size: int = 128
    shuffle: bool = True

    def __post_init__(self):
        if isinstance(self.x, np.ndarray):
            self.N_data = self.x.shape[0]
        elif isinstance(self.x, list) and all(
            isinstance(arr, np.ndarray) for arr in self.x
        ):
            self.N_data = self.x[0].shape[0]
        else:
            ic("Unsupported input type for PrecomputedMultiTaskDataGen.")
            sys.exit()
        self.indices = np.arange(self.N_data)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index: int):
        batch_ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        if isinstance(self.x, np.ndarray):
            x_batch = self.x[batch_ids]
        else:
            x_batch = [arr[batch_ids] for arr in self.x]
        y_batch = {
            "emission": self.y_emission[batch_ids],
            "plume_aux": self.y_plume[batch_ids],
        }
        return x_batch, y_batch

    def __len__(self):
        return int(np.ceil(self.N_data / self.batch_size))


def generate_beta_samples(
    shape: list,
    a: float = -30,
    b: float = 85,
    alpha: int = 8,
    beta: int = 10,
    lower_bound: float = 0,
    upper_bound: float = 45,
):
    """Generate samples from beta distribution."""
    total_samples = shape[0]
    estimated_samples_needed = int(total_samples * 1.3)
    final_samples = np.empty(shape=shape[0])
    N_valid_samples_prec = 0

    while True:
        batch_samples = np.random.beta(alpha, beta, size=estimated_samples_needed)
        scaled_batch_samples = a + (b - a) * batch_samples

        # Filter samples within the desired range
        valid_samples = scaled_batch_samples[
            (scaled_batch_samples >= lower_bound)
            & (scaled_batch_samples <= upper_bound)
        ]
        N_valid_samples = valid_samples.shape[0]
        N_samples_to_feed = np.min(
            (N_valid_samples, total_samples - N_valid_samples_prec)
        )
        final_samples[
            N_valid_samples_prec : N_samples_to_feed + N_valid_samples_prec
        ] = valid_samples[:N_samples_to_feed]
        N_valid_samples_prec = N_valid_samples

        if N_valid_samples >= total_samples:
            return final_samples.reshape(shape)
        else:
            # Increase the estimated number of samples needed for the next iteration
            estimated_samples_needed = int((total_samples - N_samples_to_feed) * 1.5)


@dataclass
class InvDataGen(tf.keras.utils.Sequence):
    """
    Custom generator to
    - produce pairs of background+s*plume and s*emissions for inversion.
    - noise data
    - normalise data
    - add clouds to data
    """

    x: Any
    xco2_plume: np.ndarray
    xco2_back: np.ndarray
    xco2_alt_anthro: np.ndarray
    no2_plume: Optional[np.ndarray]
    no2_back: Optional[np.ndarray]
    y: np.ndarray
    list_chans: list
    clouds: np.ndarray
    noise_layer: ConditionalNoiseLayer
    normalise_layer: TrainingTimeNormalization
    cloud_layer: CloudsLayer
    batch_size: int = 128
    shuffle: bool = True
    xco2_plume_scaling_min: float = 0.33
    xco2_plume_scaling_max: float = 2.25
    no2_plume_scaling_min: float = 0.75
    no2_plume_scaling_max: float = 2
    xco2_emiss_min: float = 0
    xco2_emiss_max: float = 45
    scaling_method: str = "beta_distribution_mapping"
    beta_a: float = -30
    beta_b: float = 85
    norm_noise_std_min: float = 0.0
    norm_noise_std_max: float = 0.0

    # can also be "uniform_emiss"
    def __post_init__(self):
        self.N_data = self.xco2_plume.shape[0]

        self.plume_ids = np.arange(self.N_data)
        self.back_ids = np.arange(self.N_data)
        self.xco2_alt_ids = np.arange(self.N_data)
        self.clouds_ids = np.arange(self.clouds.shape[0])

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.plume_ids)
            np.random.shuffle(self.back_ids)
            np.random.shuffle(self.xco2_alt_ids)
            np.random.shuffle(self.clouds_ids)

    def __get_input(
        self,
        plume_batches: np.ndarray,
        back_batches: np.ndarray,
        xco2_alt_batches: np.ndarray,
        cloud_batches: np.ndarray,
        xco2_plume_scaling: np.ndarray,
        back_scaling: np.ndarray,
        xco2_alt_scaling: np.ndarray,
        no2_plume_scaling: np.ndarray,
    ):
        """Get input batches with random scaling."""
        if isinstance(self.x, np.ndarray):
            x_batch = self.x[plume_batches]
        elif isinstance(self.x, list) and all(
            isinstance(arr, np.ndarray) for arr in self.x
        ):
            x_batch = self.x[0][plume_batches]
        else:
            ic("Error in generator")
            sys.exit()

        for idx, chan in enumerate(self.list_chans):
            if chan == "xco2":
                x_batch[:, :, :, idx : idx + 1] = (
                    (
                        xco2_plume_scaling[:, np.newaxis, np.newaxis, np.newaxis]
                        * self.xco2_plume[plume_batches]
                    )
                    + (
                        back_scaling[:, np.newaxis, np.newaxis, np.newaxis]
                        + self.xco2_back[back_batches]
                    )
                    + (
                        xco2_alt_scaling[:, np.newaxis, np.newaxis, np.newaxis]
                        * self.xco2_alt_anthro[xco2_alt_batches]
                    )
                )
            elif chan == "no2":
                x_batch[:, :, :, idx : idx + 1] = (
                    no2_plume_scaling.reshape(no2_plume_scaling.shape + (1,) * 3)
                    * self.no2_plume[plume_batches]
                ) + self.no2_back[back_batches]

        x_batch = self.noise_layer(x_batch, training=True)
        x_batch = self.normalise_layer(x_batch, training=True)
        x_batch = self.cloud_layer.apply_clouds_to_field(
            x_batch, self.clouds[cloud_batches]
        )
        x_batch = self.__apply_norm_noise(x_batch)

        return x_batch

    def __apply_norm_noise(self, x_batch: np.ndarray) -> np.ndarray:
        if self.norm_noise_std_max <= 0:
            return x_batch
        sigma = np.random.uniform(self.norm_noise_std_min, self.norm_noise_std_max)
        if tf.is_tensor(x_batch):
            return x_batch + tf.random.normal(
                tf.shape(x_batch),
                mean=0.0,
                stddev=sigma,
                dtype=x_batch.dtype,
            )
        return (
            x_batch
            + np.random.normal(0.0, sigma, size=x_batch.shape).astype(np.float32)
        ).astype(np.float32)

    def __get_output(self, batches: np.ndarray, plume_scaling: np.ndarray):
        """Get output batches with random scaling."""
        y_batch = plume_scaling[:, np.newaxis] * self.y[batches]

        return y_batch

    def __calculate_xco2_plume_scaling(self, plume_batches: np.ndarray):
        """Calculate xco2 plume scaling within specified bounds."""
        if self.scaling_method == "uniform_scaling":
            y_values = self.y[plume_batches]

            scaling_min = np.maximum(
                self.xco2_emiss_min / y_values, self.xco2_plume_scaling_min
            )
            scaling_max = np.minimum(
                self.xco2_emiss_max / y_values, self.xco2_plume_scaling_max
            )
            return np.squeeze(np.random.uniform(scaling_min, scaling_max))

        elif self.scaling_method == "uniform_distribution_mapping":
            y_values = self.y[plume_batches]
            random_uniform_value = np.random.uniform(
                self.xco2_emiss_min, self.xco2_emiss_max, size=y_values.shape
            )
            scaling = random_uniform_value / y_values

            return np.squeeze(scaling)

        elif self.scaling_method == "beta_distribution_mapping":
            y_values = self.y[plume_batches]
            scaled_samples = generate_beta_samples(
                y_values.shape,
                lower_bound=self.xco2_emiss_min,
                upper_bound=self.xco2_emiss_max,
                a=self.beta_a,
                b=self.beta_b,
            )
            scaling = scaled_samples / y_values
            return np.squeeze(scaling)
        else:
            ic("Typo in scaling method decleared.")
            sys.exit()

    def __get_data(
        self,
        plume_batches: np.ndarray,
        back_batches: np.ndarray,
        xco2_alt_batches: np.ndarray,
        cloud_batches: np.ndarray,
    ):
        """Get random batches, drawing random scaling."""

        xco2_plume_scaling = self.__calculate_xco2_plume_scaling(plume_batches)

        back_scaling = np.random.uniform(-3.5, 3.5, size=self.batch_size)
        xco2_alt_scaling = np.random.uniform(0.25, 3, size=self.batch_size)
        no2_plume_scaling = np.random.uniform(
            self.no2_plume_scaling_min,
            self.no2_plume_scaling_max,
            size=self.batch_size,
        )
        x_batch = self.__get_input(
            plume_batches,
            back_batches,
            xco2_alt_batches,
            cloud_batches,
            xco2_plume_scaling,
            back_scaling,
            xco2_alt_scaling,
            no2_plume_scaling,
        )

        y_batch = self.__get_output(plume_batches, xco2_plume_scaling)

        if isinstance(self.x, np.ndarray):
            return x_batch, y_batch
        elif isinstance(self.x, list) and all(
            isinstance(arr, np.ndarray) for arr in self.x
        ):
            return [x_batch, self.x[1][plume_batches]], y_batch
        else:
            sys.exit()

    def __getitem__(self, index: int):
        """Get random list of batches to draw data."""
        plume_batches = self.plume_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        back_batches = self.back_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        xco2_alt_batches = self.xco2_alt_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        cloud_batches = self.clouds_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        x, y = self.__get_data(
            plume_batches, back_batches, xco2_alt_batches, cloud_batches
        )
        return x, y

    def __len__(self):
        """Get number of batches per epoch."""
        return self.N_data // self.batch_size
