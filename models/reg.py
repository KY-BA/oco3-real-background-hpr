# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import logging
import sys
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from icecream import ic
from tensorflow import keras

from models.my_efficientnet import EfficientNet
from models.my_essential_inversors import (
    CCT,
    attention_cnn,
    active_plume_region_cnn_tiny,
    attention_feature_aux_v2_cnn_tiny,
    bg_contrastive_cnn_tiny,
    bg_plume_aux_cnn_tiny,
    cnn_afnp_tiny,
    cnn_ssm_hybrid_tiny,
    deep_cnn_advanced,
    densenet_like_model,
    essential_regressor,
    gated_cnn_tiny,
    hierarchical_plume_reasoner_tiny,
    inception_like_model,
    cnn_kan_head_tiny,
    linear_regressor,
    lk_cnn_tiny,
    moganet_tiny,
    multi_scale_residual_cnn_tiny,
    observation_set_encoder_tiny,
    point_mlp_tiny,
    res_cnn_plume_tiny,
    scale_aware_cnn_tiny,
    scale_decomposed_cnn_tiny,
    simple_resnet,
    spectral_amp_phase_cnn_tiny,
    starnet_tiny,
    supervised_plume_attention_cnn_tiny,
    unet_like_model,
    vision_ssm_tiny,
    wind_guided_residual_cnn_tiny,
)
from models.my_mobilenet import MobileNet
from models.my_shufflenet import ShuffleNet
from models.preprocessing import (
    CloudsLayer,
    ConditionalNoiseLayer,
    TrainingTimeNormalization,
)


def get_top_layers(classes: int, choice_top: str = "linear"):
    """Return top layers for regression model."""

    def top_layers(x):
        if choice_top in [
            "efficientnet",
            "squeezenet",
            "nasnet",
            "mobilenet",
            "shufflenet",
        ]:
            x = tf.keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
            x = tf.keras.layers.Dense(classes, name="regressor")(x)
            outputs = tf.keras.layers.LeakyReLU(
                alpha=0.3, dtype=tf.float32, name="regressor_activ"
            )(x)
        elif choice_top == "linear":
            outputs = tf.keras.layers.Dense(classes, name="regressor")(x)
        else:
            x = tf.keras.layers.Dense(1)(x)
            outputs = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        return outputs

    return top_layers


def get_core_model(
    name: str,
    input_shape: list,
    classes: int = 1,
    dropout_rate: float = 0.2,
):
    """Get core model for regression model."""
    if name == "efficientnet":
        core_model = EfficientNet(
            scaling_coefficient=1,
            input_shape=input_shape,
            classes=classes,
            dropout_rate=dropout_rate,
        )
    elif name == "linear":
        core_model = linear_regressor(input_shape)
    elif name == "essential":
        core_model = essential_regressor(input_shape, dropout_rate)
    elif name == "squeezenet":
        raise ValueError("SqueezeNet is not included in this manuscript release.")
    elif name == "mobilenet":
        core_model = MobileNet(input_shape, scaling_coeff=0.4)
    elif name == "shufflenet":
        core_model = ShuffleNet(input_shape, scaling_coefficient=0.75)
    elif name == "CCT":
        core_model = CCT(input_shape)
    elif name == "densenet_like_model":
        core_model = densenet_like_model(input_shape)
    elif name == "unet_like_model":
        core_model = unet_like_model(input_shape)
    elif name == "inception_like_model":
        core_model = inception_like_model(input_shape)
    elif name == "deep_cnn_advanced":
        core_model = deep_cnn_advanced(input_shape)
    elif name == "attention_cnn":
        core_model = attention_cnn(input_shape)
    elif name == "simple_resnet":
        core_model = simple_resnet(input_shape)
    elif name == "res_cnn_plume_tiny":
        core_model = res_cnn_plume_tiny(input_shape, dropout_rate)
    elif name == "vision_ssm_tiny":
        core_model = vision_ssm_tiny(input_shape, dropout_rate)
    elif name == "cnn_ssm_hybrid_tiny":
        core_model = cnn_ssm_hybrid_tiny(input_shape, dropout_rate)
    elif name in ["cnn_afnp_tiny", "cnn_afno_tiny"]:
        core_model = cnn_afnp_tiny(input_shape, dropout_rate)
    elif name == "gated_cnn_tiny":
        core_model = gated_cnn_tiny(input_shape, dropout_rate)
    elif name == "cnn_kan_head_tiny":
        core_model = cnn_kan_head_tiny(input_shape, dropout_rate)
    elif name == "point_mlp_tiny":
        core_model = point_mlp_tiny(input_shape, dropout_rate)
    elif name == "moganet_tiny":
        core_model = moganet_tiny(input_shape, dropout_rate)
    elif name == "multi_scale_residual_cnn_tiny":
        core_model = multi_scale_residual_cnn_tiny(input_shape, dropout_rate)
    elif name == "wind_guided_residual_cnn_tiny":
        core_model = wind_guided_residual_cnn_tiny(input_shape, dropout_rate)
    elif name == "hierarchical_plume_reasoner_tiny":
        core_model = hierarchical_plume_reasoner_tiny(input_shape, dropout_rate)
    elif name == "active_plume_region_cnn_tiny":
        core_model = active_plume_region_cnn_tiny(input_shape, dropout_rate)
    elif name == "lk_cnn_tiny":
        core_model = lk_cnn_tiny(input_shape, dropout_rate)
    elif name == "starnet_tiny":
        core_model = starnet_tiny(input_shape, dropout_rate)
    elif name == "spectral_amp_phase_cnn_tiny":
        core_model = spectral_amp_phase_cnn_tiny(input_shape, dropout_rate)
    elif name == "bg_contrastive_cnn_tiny":
        core_model = bg_contrastive_cnn_tiny(input_shape, dropout_rate)

    else:
        logging.info(f"Wrong model name selected: {name}")
        sys.exit()

    return core_model


@dataclass
class Reg_model_builder:
    """Return appropriate regression model."""

    name: str
    input_shape: list = field(default_factory=lambda: [64, 64, 3])
    classes: int = 1
    norm_layer: TrainingTimeNormalization = TrainingTimeNormalization(
        axis=-1, name="preproc_norm"
    )
    noise_layer: ConditionalNoiseLayer = ConditionalNoiseLayer([])
    dropout_rate: float = 0.2
    cloud_layer: CloudsLayer = CloudsLayer(np.zeros((1)), [])
    timedate_vector_size: int = 0
    set_size: int = 4

    def get_model(self):
        """Return regression model, keras or locals."""
        if self.name == "observation_set_encoder_tiny":
            return observation_set_encoder_tiny(
                self.input_shape,
                self.dropout_rate,
                self.classes,
                self.set_size,
            )
        if self.name in [
            "bg_plume_aux_cnn_tiny",
            "supervised_plume_attention_cnn_tiny",
            "attention_feature_aux_v2_cnn_tiny",
        ]:
            if self.name == "attention_feature_aux_v2_cnn_tiny":
                return attention_feature_aux_v2_cnn_tiny(
                    self.input_shape,
                    self.dropout_rate,
                    self.classes,
                )
            if self.name == "supervised_plume_attention_cnn_tiny":
                return supervised_plume_attention_cnn_tiny(
                    self.input_shape,
                    self.dropout_rate,
                    self.classes,
                )
            return bg_plume_aux_cnn_tiny(
                self.input_shape,
                self.dropout_rate,
                self.classes,
            )
        if self.name == "scale_aware_cnn_tiny":
            return scale_aware_cnn_tiny(
                self.input_shape,
                self.dropout_rate,
                self.classes,
            )
        if self.name == "scale_decomposed_cnn_tiny":
            return scale_decomposed_cnn_tiny(
                self.input_shape,
                self.dropout_rate,
                self.classes,
            )

        self.core_model = get_core_model(
            self.name,
            self.input_shape,
            self.classes,
            self.dropout_rate,
        )
        self.top_layers = get_top_layers(self.classes, self.name)

        inputs = tf.keras.layers.Input(self.input_shape, name="input_layer")
        x = self.core_model(inputs)

        if self.timedate_vector_size > 0:
            timedate_input = tf.keras.layers.Input(
                shape=self.timedate_vector_size, name="timedate_input_layer"
            )
            concatenated = tf.keras.layers.Concatenate()([x, timedate_input])
            x = tf.keras.layers.Dense(16, activation="elu")(concatenated)
            outputs = self.top_layers(x)

        else:
            outputs = self.top_layers(x)

        model = tf.keras.Model(inputs, outputs)
        return model
