# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import sys
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from icecream import ic


class TrainingTimeNormalization(tf.keras.layers.Normalization):
    """Normalisation layer."""

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        if training:
            return super().call(inputs)
        else:
            return inputs

    def get_config(self):
        config = super().get_config()
        return config


class CloudsLayer(tf.keras.layers.Layer):
    """Specific layer to add clouds to images only at training."""

    def __init__(
        self, clouds_array=[], list_chans=[], nanmin=[], nanmedian=[], **kwargs
    ):
        super().__init__(**kwargs)
        self.clouds_array = clouds_array
        self.list_chans = list_chans
        self.eval_clouds_array = None
        self.nanmin = nanmin
        self.nanmedian = nanmedian

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "list_chans": self.list_chans,
                "nanmin": self.nanmin.numpy().tolist(),
                "nanmedian": self.nanmedian.numpy().tolist(),
            }
        )
        return config

    def evaluate_nanvalues(self, train_data):
        """Evaluate nan min and nan median values on train_data."""
        nanmin_np = np.nanmin(train_data, axis=(0, 1, 2))
        nanmedian_np = np.nanmedian(train_data, axis=(0, 1, 2))
        self.nanmin = tf.convert_to_tensor(nanmin_np, dtype=tf.float32)
        self.nanmedian = tf.convert_to_tensor(nanmedian_np, dtype=tf.float32)

    def call(self, inputs, training=None, **kwargs):
        if training:
            random_indices = np.random.choice(
                self.clouds_array.shape[0], size=inputs.shape[0], replace=True
            )
            selected_clouds = self.clouds_array[random_indices]
            clouded_fields = self.apply_clouds_to_field(inputs, selected_clouds)
            return clouded_fields
        else:
            return inputs

    def apply_clouds_to_field(self, field, clouds):
        """Apply clouds to field given list_chans.
        Accepts clouds with shape (B,H,W) or (B,H,W,1).
        """
        # Ensure clouds has (B,H,W) to match per-channel data shape (B,H,W)
        clouds_tensor = tf.convert_to_tensor(clouds)
        if len(clouds_tensor.shape) == 4 and clouds_tensor.shape[-1] == 1:
            clouds_tensor = tf.squeeze(clouds_tensor, axis=-1)

        dtype = field.dtype
        list_tensor = [None] * len(self.list_chans)
        for idx, chan_name in enumerate(self.list_chans):
            if chan_name in ["xco2", "no2"]:
                threshold = 0.01 if chan_name == "xco2" else 0.3
                channel_data = field[:, :, :, idx]

                # Align clouds spatial size to channel_data if needed
                ch_shape = tf.shape(channel_data)
                H, W = ch_shape[1], ch_shape[2]
                cl_shape = tf.shape(clouds_tensor)
                cH, cW = cl_shape[1], cl_shape[2]
                clouds_aligned = tf.cond(
                    tf.logical_or(tf.not_equal(H, cH), tf.not_equal(W, cW)),
                    lambda: tf.squeeze(
                        tf.image.resize(
                            tf.expand_dims(clouds_tensor, -1),
                            size=(H, W),
                            method="nearest",
                            preserve_aspect_ratio=False,
                        ),
                        axis=-1,
                    ),
                    lambda: clouds_tensor,
                )

                bin_threshold = tf.cast(clouds_aligned < threshold, dtype=dtype)

                modified_channel_data = tf.where(
                    tf.equal(bin_threshold, 0),
                    tf.cast(tf.constant(np.nan), dtype),
                    channel_data,
                )

                # Determine the appropriate numpy function based on chan_name
                nan_val = self.nanmin if chan_name == "xco2" else self.nanmedian

                # Apply the function and replace NaNs in the new tensor
                nan_fill = tf.fill(
                    tf.shape(modified_channel_data),
                    tf.cast(nan_val[idx], dtype),
                )

                list_tensor[idx] = tf.where(
                    tf.math.is_nan(modified_channel_data),
                    nan_fill,
                    modified_channel_data,
                )
            else:
                list_tensor[idx] = field[:, :, :, idx]
        new_field = tf.stack(list_tensor, axis=-1)
        return new_field


class ConditionalNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, list_chans, xco2_noise=0.7, no2_noise=1e15, **kwargs):
        super().__init__(**kwargs)
        self.list_chans = list_chans
        self.xco2_noise = xco2_noise
        self.no2_noise = no2_noise

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "list_chans": self.list_chans,
                "stddev": self.xco2_noise,
            }
        )
        return config

    def call(self, inputs, training=None):
        if training:
            outputs = [None] * len(self.list_chans)
            for idx in range(len(self.list_chans)):
                if self.list_chans[idx] == "xco2":
                    outputs[idx] = tf.keras.layers.GaussianNoise(
                        stddev=self.xco2_noise, name=f"xco2_noise_{idx}"
                    )(inputs[:, :, :, idx : idx + 1])
                elif self.list_chans[idx] == "no2":
                    outputs[idx] = tf.keras.layers.GaussianNoise(
                        stddev=self.no2_noise, name=f"no2_noise_{idx}"
                    )(inputs[:, :, :, idx : idx + 1])

                else:
                    outputs[idx] = tf.keras.layers.Layer()(
                        inputs[:, :, :, idx : idx + 1]
                    )
            return tf.keras.layers.Concatenate()(outputs)
        else:
            return inputs
