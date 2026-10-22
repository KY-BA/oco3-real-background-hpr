# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import sys

import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow import keras


def pixel_weighted_cross_entropy(
    y_true: tf.Tensor, y_pred: tf.Tensor, reduction: bool = True
):
    """Loss function for segmentation weighted.
    Loss for weighting each pixel loss with y_true[pixel] value."""
    y_bin_true = tf.cast(y_true > 0, y_true.dtype)
    loss_val = keras.losses.binary_crossentropy(y_bin_true, y_pred)
    weights = tf.where(y_true > 0, y_true, 1.0)
    loss_val = tf.convert_to_tensor(tf.squeeze(weights), float) * loss_val
    if reduction:
        return K.mean(loss_val)
    if not reduction:
        return loss_val


def extreme_weighted_msle(y_mean, y_min, y_max):
    """Loss function for regression.
    Add loss weight to low and high labels in order to avoid mean regression."""
    w_max = 2
    w_min = 0.5

    a_min_mean = (w_max - w_min) / (y_min - y_mean)
    b_min_mean = w_max - a_min_mean * y_min

    a_max_mean = (w_max - w_min) / (y_max - y_mean)
    b_max_mean = w_max - a_max_mean * y_max

    def loss(y_true, y_pred):
        if y_true < y_mean:
            weight = a_min_mean * y_true + b_min_mean
            return weight * tf.keras.losses.MeanSquaredLogarithmicError(y_true, y_pred)
        elif y_true > y_mean:
            weight = a_max_mean * y_true + b_max_mean
            return weight * tf.keras.losses.MeanSquaredLogarithmicError(y_true, y_pred)

    return loss


def double_relative_error(y_true, y_pred):
    """Return relative mean absolute error."""
    return tf.math.reduce_mean((y_true - y_pred) ** 2 / (y_true * y_pred))


@tf.keras.utils.register_keras_serializable(package="CO2")
class CounterfactualScaleConsistentMAE(tf.keras.losses.Loss):
    """MAE plus adjacent-pair background consistency and asymmetric tail penalty."""

    def __init__(
        self,
        cf_weight: float = 0.2,
        tail_weight: float = 0.05,
        high_threshold: float = 18.0,
        low_threshold: float = 8.0,
        name: str = "CounterfactualScaleConsistentMAE",
    ):
        super().__init__(name=name)
        self.cf_weight = cf_weight
        self.tail_weight = tail_weight
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def call(self, y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

        mae = tf.reduce_mean(tf.abs(y_true - y_pred))

        n_pairs = tf.shape(y_pred)[0] // 2
        paired_pred = tf.reshape(y_pred[: n_pairs * 2], [n_pairs, 2])
        cf_loss = tf.cond(
            n_pairs > 0,
            lambda: tf.reduce_mean(tf.abs(paired_pred[:, 0] - paired_pred[:, 1])),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

        high_mask = tf.cast(y_true >= self.high_threshold, tf.float32)
        low_mask = tf.cast(y_true <= self.low_threshold, tf.float32)
        high_under = high_mask * tf.nn.relu(y_true - y_pred)
        low_over = low_mask * tf.nn.relu(y_pred - y_true)
        tail_loss = tf.reduce_mean(high_under + low_over)

        return mae + self.cf_weight * cf_loss + self.tail_weight * tail_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "cf_weight": self.cf_weight,
                "tail_weight": self.tail_weight,
                "high_threshold": self.high_threshold,
                "low_threshold": self.low_threshold,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="CO2")
class ScaleAwarePerturbationConsistentMAE(tf.keras.losses.Loss):
    """MAE with clean/noisy prediction consistency and low-emission overestimate guard."""

    def __init__(
        self,
        consistency_weight: float = 0.2,
        low_weight: float = 0.1,
        low_margin: float = 0.0,
        name: str = "ScaleAwarePerturbationConsistentMAE",
    ):
        super().__init__(name=name)
        self.consistency_weight = consistency_weight
        self.low_weight = low_weight
        self.low_margin = low_margin

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

        y_rank = tf.rank(y_true)
        n_cols = tf.cond(
            tf.equal(y_rank, 1),
            lambda: tf.constant(1, dtype=tf.int32),
            lambda: tf.shape(y_true)[1],
        )
        y_2d = tf.cond(
            tf.equal(y_rank, 1),
            lambda: tf.reshape(y_true, [-1, 1]),
            lambda: y_true,
        )
        y_label = tf.reshape(y_2d[:, 0], [-1])
        mae = tf.reduce_mean(tf.abs(y_label - y_pred))

        def paired_loss():
            n_total = tf.shape(y_pred)[0]
            n_pair = n_total // 2
            clean_pred = y_pred[:n_pair]
            noisy_pred = y_pred[n_pair : n_pair * 2]
            clean_y = y_label[:n_pair]
            low_flag = y_2d[:n_pair, 1]
            consistency = tf.reduce_mean(tf.abs(clean_pred - noisy_pred))
            low_over = tf.nn.relu(noisy_pred - clean_y - self.low_margin)
            low_loss = tf.math.divide_no_nan(
                tf.reduce_sum(low_flag * low_over),
                tf.reduce_sum(low_flag),
            )
            return mae + self.consistency_weight * consistency + self.low_weight * low_loss

        return tf.cond(
            n_cols >= 2,
            paired_loss,
            lambda: mae,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "consistency_weight": self.consistency_weight,
                "low_weight": self.low_weight,
                "low_margin": self.low_margin,
            }
        )
        return config


def define_loss(name_loss):
    """Return appropriate loss function."""
    call_dict = {
        "pixel_weighted_cross_entropy": pixel_weighted_cross_entropy,
        "BinaryCrossentropy": tf.keras.losses.BinaryCrossentropy(),
        "MeanSquaredLogarithmicError": tf.keras.losses.MeanSquaredLogarithmicError(),
        "MeanAbsolutePercentageError": tf.keras.losses.MeanAbsolutePercentageError(),
        "MeanSquaredError": tf.keras.losses.MeanSquaredError(),
        "MeanAbsoluteError": tf.keras.losses.MeanAbsoluteError(),
        "CounterfactualScaleConsistentMAE": CounterfactualScaleConsistentMAE(),
        "CounterfactualConsistentMAE": CounterfactualScaleConsistentMAE(
            cf_weight=0.2,
            tail_weight=0.0,
            name="CounterfactualConsistentMAE",
        ),
        "CounterfactualScaleConsistentMAEWeak": CounterfactualScaleConsistentMAE(
            cf_weight=0.1,
            tail_weight=0.02,
            name="CounterfactualScaleConsistentMAEWeak",
        ),
        "ScaleAwarePerturbationConsistentMAE": ScaleAwarePerturbationConsistentMAE(),
    }
    loss = call_dict[name_loss]
    return loss


def define_metrics(exp_purpose: str):
    """Return list of metric functions."""
    metrics = []
    if exp_purpose == "segmentation":
        metrics = []
    elif exp_purpose == "inversion":
        metrics = [
            tf.keras.losses.MeanAbsolutePercentageError(),
            tf.keras.losses.MeanAbsoluteError(),
        ]
    return metrics


def calculate_weighted_plume(
    plume: np.ndarray,
    min_w: float,
    max_w: float,
    curve: str = "linear",
    param_curve: float = 1,
):
    """Calculate a weighted plume given min_w, max_w, and a curve between the two."""

    N_data = plume.shape[0]
    y_min = np.repeat(
        [np.where(plume > 0, plume, np.max(plume)).min()], N_data
    ).reshape(N_data, 1, 1)
    y_max = np.quantile(plume, q=0.995, axis=(1, 2)).reshape(N_data, 1, 1)
    weight_min = np.repeat([min_w], N_data).reshape(N_data, 1, 1)
    weight_max = np.repeat([max_w], N_data).reshape(N_data, 1, 1)

    if curve == "linear":
        pente = (weight_max - weight_min) / (y_max - y_min)  # type: ignore
        b = weight_min - pente * y_min
        y_data = pente * plume + b * np.where(plume > 0, 1, 0)

    elif curve == "exponential":
        A_0 = (weight_min - weight_max) / (  # type: ignore
            np.exp(param_curve * y_min) - np.exp(param_curve * y_max)
        )
        b = (
            weight_min
            + weight_max
            - A_0 * (np.exp(param_curve * y_min) + np.exp(param_curve * y_max))
        ) / 2
        y_data = A_0 * np.exp(param_curve * plume) + b * np.where(plume > 0, 1, 0)
        y_data = np.where(plume > 0, y_data, 0)
    else:
        sys.exit()

    y_data = np.where(y_data < max_w, y_data, max_w)
    y_data = np.expand_dims(y_data, axis=-1)

    return y_data
