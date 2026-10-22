# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    LayerNormalization,
    MaxPooling2D,
    MultiHeadAttention,
    Multiply,
    Reshape,
    Subtract,
    concatenate,
)
from tensorflow.keras.models import Model


def linear_regressor(input_shape: list):
    """Linear regressor."""
    core_model = tf.keras.models.Sequential()
    core_model.add(tf.keras.Input(shape=input_shape))
    core_model.add(tf.keras.layers.Conv2D(16, (3, 3)))
    core_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    core_model.add(tf.keras.layers.Conv2D(16, (3, 3)))
    core_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    core_model.add(tf.keras.layers.Conv2D(16, (3, 3)))
    core_model.add(tf.keras.layers.Flatten())
    core_model.add(tf.keras.layers.Dense(16))
    return core_model


def essential_regressor(input_shape: list, dropout_rate: float = 0.2):
    """Essential regressor."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(inputs)
    x = tf.keras.layers.Dropout(dropout_rate / 2)(x)
    if input_shape[1] == 64:
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate * 3 / 2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate * 3 / 2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    core_model = tf.keras.Model(inputs, x)
    return core_model


def residual_block(x, filters, kernel_size=(3, 3), strides=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding="same", strides=strides)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding="same")(
        shortcut
    )
    shortcut = tf.keras.layers.Dropout(0.3)(shortcut)
    x = Add()([x, shortcut])
    x = Activation("elu")(x)
    return x


def simple_resnet(input_shape):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64, strides=2)
    x = residual_block(x, 64)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = Flatten()(x)
    model = Model(inputs, x)
    return model


def plume_residual_block(x, filters, dropout_rate=0.15, name="plume_res_block"):
    """Small residual block tuned for 32x32 plume-regression fields."""
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same", name=f"{name}_shortcut")(
            shortcut
        )

    y = Conv2D(filters, (3, 3), padding="same", activation="elu", name=f"{name}_conv_1")(
        x
    )
    y = BatchNormalization(name=f"{name}_bn_1")(y)
    y = Dropout(dropout_rate, name=f"{name}_dropout")(y)
    y = Conv2D(filters, (3, 3), padding="same", name=f"{name}_conv_2")(y)
    y = BatchNormalization(name=f"{name}_bn_2")(y)
    y = Add(name=f"{name}_add")([shortcut, y])
    return Activation("elu", name=f"{name}_activation")(y)


def res_cnn_plume_tiny(input_shape, dropout_rate=0.2):
    """Task-tuned residual CNN for small XCO2 plume regression fields."""
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="plume_stem_conv_1")(
        inputs
    )
    x = BatchNormalization(name="plume_stem_bn_1")(x)
    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="plume_stem_conv_2")(
        x
    )
    x = BatchNormalization(name="plume_stem_bn_2")(x)

    x = plume_residual_block(
        x,
        32,
        dropout_rate=dropout_rate / 2,
        name="plume_stage_1_block_1",
    )
    x = plume_residual_block(
        x,
        32,
        dropout_rate=dropout_rate / 2,
        name="plume_stage_1_block_2",
    )
    x = MaxPooling2D((2, 2), name="plume_downsample_1")(x)

    x = plume_residual_block(
        x,
        64,
        dropout_rate=dropout_rate,
        name="plume_stage_2_block_1",
    )
    x = plume_residual_block(
        x,
        64,
        dropout_rate=dropout_rate,
        name="plume_stage_2_block_2",
    )
    x = MaxPooling2D((2, 2), name="plume_downsample_2")(x)

    x = plume_residual_block(
        x,
        96,
        dropout_rate=dropout_rate,
        name="plume_stage_3_block_1",
    )
    x = plume_residual_block(
        x,
        96,
        dropout_rate=dropout_rate,
        name="plume_stage_3_block_2",
    )

    x = plume_residual_block(
        x,
        128,
        dropout_rate=dropout_rate,
        name="plume_stage_4_block_1",
    )

    avg_pool = GlobalAveragePooling2D(name="plume_avg_pool")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D(name="plume_max_pool")(x)
    x = Concatenate(name="plume_pool_concat")([avg_pool, max_pool])
    x = Dense(96, activation="elu", name="plume_head_dense_1")(x)
    x = Dropout(dropout_rate, name="plume_head_dropout")(x)
    x = Dense(64, activation="elu", name="plume_head_dense_2")(x)
    return Model(inputs, x)


def attention_module(x, filters):
    p = 1  # Pooling size
    t = 2  # Transformation ratio
    r = 1  # Number of attention modules

    x = Conv2D(filters * t, kernel_size=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = Conv2D(filters, kernel_size=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)

    return x


def attention_cnn(input_shape):
    inputs = Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    attention = attention_module(x, 32)
    attention = attention_module(attention, 32)
    x = Multiply()([x, attention])
    x = tf.keras.layers.MaxPooling2D()(x)

    attention = attention_module(x, 64)
    attention = attention_module(attention, 64)
    x = Multiply()([x, attention])
    x = tf.keras.layers.MaxPooling2D()(x)

    attention = attention_module(x, 64)
    attention = attention_module(attention, 64)
    x = Multiply()([x, attention])
    x = tf.keras.layers.MaxPooling2D()(x)

    x = Flatten()(x)
    model = Model(inputs, x)
    return model


def deep_cnn_advanced(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="elu", strides=1)(inputs)
    x = Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation="elu", strides=1)(x)
    x = Conv2D(128, (3, 3), activation="elu", strides=1)(x)
    x = Dropout(0.4)(x)
    x = MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.AveragePooling2D()(x)
    x = Flatten()(x)
    model = Model(inputs, x)
    return model


def inception_module(x, filters):
    branch1 = Conv2D(filters, (1, 1), activation="elu", padding="same")(x)

    branch2 = Conv2D(filters, (1, 1), activation="elu", padding="same")(x)
    branch2 = Conv2D(filters, (3, 3), activation="elu", padding="same")(branch2)

    branch3 = Conv2D(filters, (1, 1), activation="elu", padding="same")(x)
    branch3 = Conv2D(filters, (5, 5), activation="elu", padding="same")(branch3)

    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch4 = Conv2D(filters, (1, 1), activation="elu", padding="same")(branch4)

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    return x


def inception_like_model(input_shape):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = inception_module(inputs, 16)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = MaxPooling2D((2, 2))(x)

    x = inception_module(x, 32)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.AveragePooling2D()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.AveragePooling2D()(x)
    x = Flatten()(x)
    model = Model(inputs, x)
    return model


def conv_block(x, filters):
    x = Conv2D(filters, (3, 3), activation="elu", padding="same")(x)
    x = Conv2D(filters, (3, 3), activation="elu", padding="same")(x)
    return x


def unet_like_model(input_shape):
    inputs = Input(shape=input_shape)

    c1 = conv_block(inputs, 16)  # Reduced from 32 to 16
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 32)  # Reduced from 64 to 32
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 64)  # Reduced from 128 to 64

    u4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c3)
    u4 = concatenate([u4, c2])
    c4 = conv_block(u4, 32)

    u5 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c4)
    u5 = concatenate([u5, c1])
    c5 = conv_block(u5, 8)

    x = Conv2D(32, (3, 3), activation="elu", padding="same")(
        c5
    )  # Adjust the number of filters

    # Average Pooling to reduce spatial dimensions
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    model = Model(inputs, x)
    return model


def dense_block(x, growth_rate, layers_in_block):
    for _ in range(layers_in_block):
        cb = Conv2D(growth_rate, (3, 3), padding="same")(x)
        cb = Dropout(0.2)(cb)
        cb = BatchNormalization()(cb)
        cb = Activation("elu")(cb)
        x = Concatenate(axis=-1)([x, cb])
    return x


def transition_layer(x, compression_factor=0.5):
    reduced_filters = int(x.shape[-1] * compression_factor)
    x = Conv2D(reduced_filters, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((2, 2))(x)
    return x


def densenet_like_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (7, 7), strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = dense_block(x, 16, 3)
    x = transition_layer(x)

    x = dense_block(x, 16, 3)
    x = transition_layer(x)

    x = dense_block(x, 32, 3)
    x = transition_layer(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)

    x = Flatten()(x)

    model = Model(inputs, x)
    return model


def ssm_token_mixer_block(
    x,
    embed_dim,
    mlp_ratio=2,
    dropout_rate=0.2,
    kernel_size=9,
    name="ssm_block",
):
    """Lightweight gated sequence mixer inspired by Vision-SSM/Mamba blocks."""
    residual = x
    y = LayerNormalization(epsilon=1e-5, name=f"{name}_token_norm")(x)
    value = Dense(embed_dim, activation="elu", name=f"{name}_value_proj")(y)
    gate = Dense(embed_dim, activation="sigmoid", name=f"{name}_gate_proj")(y)
    y = tf.keras.layers.Conv1D(
        embed_dim,
        kernel_size,
        padding="same",
        activation="elu",
        name=f"{name}_long_token_mixer",
    )(value)
    y = Dropout(dropout_rate, name=f"{name}_token_dropout")(y)
    y = Multiply(name=f"{name}_gated_mixer")([y, gate])
    x = Add(name=f"{name}_token_residual")([residual, y])

    residual = x
    y = LayerNormalization(epsilon=1e-5, name=f"{name}_ffn_norm")(x)
    y = Dense(embed_dim * mlp_ratio, activation="elu", name=f"{name}_ffn_1")(y)
    y = Dropout(dropout_rate, name=f"{name}_ffn_dropout_1")(y)
    y = Dense(embed_dim, name=f"{name}_ffn_2")(y)
    y = Dropout(dropout_rate, name=f"{name}_ffn_dropout_2")(y)
    x = Add(name=f"{name}_ffn_residual")([residual, y])
    return x


def vision_ssm_tiny(
    input_shape,
    dropout_rate=0.2,
    embed_dim=64,
    depth=4,
    patch_size=4,
    kernel_size=9,
    mlp_ratio=2,
):
    """Tiny Keras-native Vision-SSM/Mamba-style regressor backbone."""
    inputs = Input(shape=input_shape)
    x = Conv2D(
        embed_dim,
        patch_size,
        strides=patch_size,
        padding="same",
        name="ssm_patch_embed",
    )(inputs)
    x = BatchNormalization(name="ssm_patch_bn")(x)
    x = Activation("elu", name="ssm_patch_activation")(x)

    tokens_h = int(np.ceil(input_shape[0] / patch_size))
    tokens_w = int(np.ceil(input_shape[1] / patch_size))
    x = Reshape((tokens_h * tokens_w, embed_dim), name="ssm_tokens")(x)

    for i in range(depth):
        x = ssm_token_mixer_block(
            x,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            kernel_size=kernel_size,
            name=f"ssm_block_{i + 1}",
        )

    x = LayerNormalization(epsilon=1e-5, name="ssm_final_norm")(x)
    x = GlobalAveragePooling1D(name="ssm_sequence_pool")(x)
    model = Model(inputs, x)
    return model


def cnn_ssm_hybrid_tiny(
    input_shape,
    dropout_rate=0.2,
    embed_dim=64,
    depth=3,
    kernel_size=9,
    mlp_ratio=2,
):
    """Hybrid local CNN stem plus lightweight SSM-style token mixer."""
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="hybrid_stem_conv_1")(
        inputs
    )
    x = BatchNormalization(name="hybrid_stem_bn_1")(x)
    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="hybrid_stem_conv_2")(
        x
    )
    x = BatchNormalization(name="hybrid_stem_bn_2")(x)
    x = MaxPooling2D((2, 2), name="hybrid_stem_pool_1")(x)

    x = Conv2D(64, (3, 3), padding="same", activation="elu", name="hybrid_stem_conv_3")(
        x
    )
    x = Dropout(dropout_rate, name="hybrid_stem_dropout_1")(x)
    x = BatchNormalization(name="hybrid_stem_bn_3")(x)
    x = Conv2D(64, (3, 3), padding="same", activation="elu", name="hybrid_stem_conv_4")(
        x
    )
    x = BatchNormalization(name="hybrid_stem_bn_4")(x)
    x = MaxPooling2D((2, 2), name="hybrid_stem_pool_2")(x)

    x = Conv2D(embed_dim, (1, 1), padding="same", activation="elu", name="hybrid_embed")(
        x
    )
    tokens_h = int(np.ceil(input_shape[0] / 4))
    tokens_w = int(np.ceil(input_shape[1] / 4))
    x = Reshape((tokens_h * tokens_w, embed_dim), name="hybrid_tokens")(x)

    for i in range(depth):
        x = ssm_token_mixer_block(
            x,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            kernel_size=kernel_size,
            name=f"hybrid_ssm_block_{i + 1}",
        )

    x = LayerNormalization(epsilon=1e-5, name="hybrid_final_norm")(x)
    x = GlobalAveragePooling1D(name="hybrid_sequence_pool")(x)
    model = Model(inputs, x)
    return model


def fft2d_real_imag(x):
    """Apply FFT2D over spatial dimensions and expose real/imag channels."""
    x_complex = tf.cast(x, tf.complex64)
    x_complex = tf.transpose(x_complex, [0, 3, 1, 2])
    spectrum = tf.signal.fft2d(x_complex)
    spectrum = tf.transpose(spectrum, [0, 2, 3, 1])
    return tf.concat([tf.math.real(spectrum), tf.math.imag(spectrum)], axis=-1)


def ifft2d_from_real_imag(x):
    """Invert a real/imag channel representation back to spatial features."""
    n_channels = tf.shape(x)[-1] // 2
    real = x[..., :n_channels]
    imag = x[..., n_channels:]
    spectrum = tf.complex(real, imag)
    spectrum = tf.transpose(spectrum, [0, 3, 1, 2])
    spatial = tf.signal.ifft2d(spectrum)
    spatial = tf.transpose(spatial, [0, 2, 3, 1])
    return tf.math.real(spatial)


def afno_mixer_block(x, channels, dropout_rate=0.2, name="afno_block"):
    """Small FFT-domain residual mixer inspired by AFNO/FNO blocks."""
    residual = x
    y = LayerNormalization(epsilon=1e-5, name=f"{name}_freq_norm")(x)
    y = Lambda(fft2d_real_imag, name=f"{name}_fft2d_real_imag")(y)
    y = Conv2D(
        2 * channels,
        (1, 1),
        activation="elu",
        padding="same",
        name=f"{name}_freq_mix_1",
    )(y)
    y = Dropout(dropout_rate, name=f"{name}_freq_dropout")(y)
    y = Conv2D(2 * channels, (1, 1), padding="same", name=f"{name}_freq_mix_2")(y)
    y = Lambda(ifft2d_from_real_imag, name=f"{name}_ifft2d")(y)
    x = Add(name=f"{name}_freq_residual")([residual, y])

    residual = x
    y = LayerNormalization(epsilon=1e-5, name=f"{name}_ffn_norm")(x)
    y = Conv2D(
        2 * channels,
        (1, 1),
        activation="elu",
        padding="same",
        name=f"{name}_ffn_1",
    )(y)
    y = Dropout(dropout_rate, name=f"{name}_ffn_dropout")(y)
    y = Conv2D(channels, (1, 1), padding="same", name=f"{name}_ffn_2")(y)
    x = Add(name=f"{name}_ffn_residual")([residual, y])
    return x


def cnn_afnp_tiny(
    input_shape,
    dropout_rate=0.2,
    channels=64,
    depth=3,
):
    """CNN stem plus AFNO/FNO-style frequency-domain mixer for regression."""
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="afnp_stem_conv_1")(
        inputs
    )
    x = BatchNormalization(name="afnp_stem_bn_1")(x)
    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="afnp_stem_conv_2")(
        x
    )
    x = BatchNormalization(name="afnp_stem_bn_2")(x)
    x = MaxPooling2D((2, 2), name="afnp_stem_pool_1")(x)

    x = Conv2D(64, (3, 3), padding="same", activation="elu", name="afnp_stem_conv_3")(
        x
    )
    x = Dropout(dropout_rate, name="afnp_stem_dropout_1")(x)
    x = BatchNormalization(name="afnp_stem_bn_3")(x)
    x = Conv2D(
        channels,
        (3, 3),
        padding="same",
        activation="elu",
        name="afnp_stem_conv_4",
    )(x)
    x = BatchNormalization(name="afnp_stem_bn_4")(x)
    x = MaxPooling2D((2, 2), name="afnp_stem_pool_2")(x)

    for i in range(depth):
        x = afno_mixer_block(
            x,
            channels=channels,
            dropout_rate=dropout_rate,
            name=f"afnp_block_{i + 1}",
        )

    x = LayerNormalization(epsilon=1e-5, name="afnp_final_norm")(x)
    x = GlobalAveragePooling2D(name="afnp_global_pool")(x)
    model = Model(inputs, x)
    return model


def gated_conv_block(x, filters, dropout_rate=0.2, name="gated_conv_block"):
    """Residual gated CNN block for small spatial regression fields."""
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same", name=f"{name}_shortcut")(shortcut)

    y = LayerNormalization(epsilon=1e-5, name=f"{name}_norm_1")(x)
    y = Conv2D(filters, (3, 3), padding="same", activation="elu", name=f"{name}_spatial_conv")(y)
    value = Conv2D(filters, (1, 1), padding="same", activation="elu", name=f"{name}_value")(y)
    gate = Conv2D(filters, (1, 1), padding="same", activation="sigmoid", name=f"{name}_gate")(y)
    y = Multiply(name=f"{name}_gated_features")([value, gate])
    y = Dropout(dropout_rate, name=f"{name}_dropout_1")(y)
    x = Add(name=f"{name}_residual_1")([shortcut, y])

    shortcut = x
    y = LayerNormalization(epsilon=1e-5, name=f"{name}_norm_2")(x)
    y = Conv2D(2 * filters, (1, 1), padding="same", activation="elu", name=f"{name}_ffn_1")(y)
    y = Dropout(dropout_rate, name=f"{name}_dropout_2")(y)
    y = Conv2D(filters, (1, 1), padding="same", name=f"{name}_ffn_2")(y)
    y = Dropout(dropout_rate, name=f"{name}_dropout_3")(y)
    x = Add(name=f"{name}_residual_2")([shortcut, y])
    return x


def gated_cnn_tiny(input_shape, dropout_rate=0.2):
    """Modern gated CNN backbone inspired by recent MambaOut-style vision blocks."""
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="gated_stem_conv")(inputs)
    x = BatchNormalization(name="gated_stem_bn")(x)
    x = gated_conv_block(x, 32, dropout_rate=dropout_rate, name="gated_block_1")
    x = MaxPooling2D((2, 2), name="gated_pool_1")(x)

    x = Conv2D(64, (3, 3), padding="same", activation="elu", name="gated_stage_2_conv")(x)
    x = BatchNormalization(name="gated_stage_2_bn")(x)
    x = gated_conv_block(x, 64, dropout_rate=dropout_rate, name="gated_block_2")
    x = MaxPooling2D((2, 2), name="gated_pool_2")(x)

    x = Conv2D(96, (3, 3), padding="same", activation="elu", name="gated_stage_3_conv")(x)
    x = BatchNormalization(name="gated_stage_3_bn")(x)
    x = gated_conv_block(x, 96, dropout_rate=dropout_rate, name="gated_block_3")

    x = Conv2D(128, (3, 3), padding="same", activation="elu", name="gated_stage_4_conv")(x)
    x = BatchNormalization(name="gated_stage_4_bn")(x)
    x = gated_conv_block(x, 128, dropout_rate=dropout_rate, name="gated_block_4")

    x = LayerNormalization(epsilon=1e-5, name="gated_final_norm")(x)
    x = GlobalAveragePooling2D(name="gated_global_pool")(x)
    x = Dense(64, activation="elu", name="gated_feature_head")(x)
    x = Dropout(dropout_rate, name="gated_feature_dropout")(x)
    model = Model(inputs, x)
    return model


def kan_like_dense_features(x, units, name="kan_like_dense"):
    """KAN-style fixed basis expansion followed by a trainable projection."""
    linear = Dense(units, name=f"{name}_linear")(x)
    sine = Lambda(lambda t: tf.sin(t), name=f"{name}_sin_activation")(
        Dense(units, name=f"{name}_sin_proj")(x)
    )
    cosine = Lambda(lambda t: tf.cos(t), name=f"{name}_cos_activation")(
        Dense(units, name=f"{name}_cos_proj")(x)
    )
    squared = Dense(units, name=f"{name}_square_proj")(tf.square(x))
    mixed = Concatenate(name=f"{name}_basis_concat")([linear, sine, cosine, squared])
    return Dense(units, activation="elu", name=f"{name}_basis_mix")(mixed)


def cnn_kan_head_tiny(input_shape, dropout_rate=0.2):
    """CNN feature extractor with a KAN-like nonlinear regression head."""
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="elu", strides=1, name="kan_stem_conv_1")(inputs)
    x = Dropout(dropout_rate / 2, name="kan_stem_dropout_1")(x)
    if input_shape[1] == 64:
        x = Conv2D(32, (3, 3), activation="elu", strides=1, name="kan_stem_conv_2")(x)
        x = MaxPooling2D((2, 2), padding="valid", strides=2, name="kan_stem_pool_1")(x)
    x = BatchNormalization(name="kan_stem_bn_1")(x)

    x = Conv2D(32, (3, 3), activation="elu", strides=1, name="kan_stage_2_conv_1")(x)
    x = Dropout(dropout_rate, name="kan_stage_2_dropout")(x)
    x = Conv2D(64, (3, 3), activation="elu", strides=1, name="kan_stage_2_conv_2")(x)
    x = BatchNormalization(name="kan_stage_2_bn")(x)

    x = Conv2D(64, (3, 3), activation="elu", strides=1, name="kan_stage_3_conv_1")(x)
    x = Dropout(dropout_rate * 3 / 2, name="kan_stage_3_dropout")(x)
    x = Conv2D(64, (3, 3), activation="elu", strides=1, name="kan_stage_3_conv_2")(x)
    x = MaxPooling2D((2, 2), padding="valid", strides=2, name="kan_stage_3_pool")(x)

    x = Conv2D(64, (3, 3), activation="elu", strides=1, name="kan_stage_4_conv_1")(x)
    x = Dropout(dropout_rate * 3 / 2, name="kan_stage_4_dropout")(x)
    x = Conv2D(64, (3, 3), activation="elu", strides=1, name="kan_stage_4_conv_2")(x)
    x = MaxPooling2D((2, 2), padding="valid", strides=2, name="kan_stage_4_pool")(x)

    x = Flatten(name="kan_flatten")(x)
    x = Dense(128, activation="elu", name="kan_pre_head")(x)
    x = Dropout(dropout_rate, name="kan_pre_head_dropout")(x)
    x = kan_like_dense_features(x, 64, name="kan_head_1")
    x = Dropout(dropout_rate, name="kan_head_dropout_1")(x)
    x = kan_like_dense_features(x, 32, name="kan_head_2")
    model = Model(inputs, x)
    return model


def add_normalized_grid_coordinates(x):
    """Convert a 2D field to point tokens with fixed normalized row/col coordinates."""
    batch_size = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    channels = tf.shape(x)[3]

    row = tf.linspace(-1.0, 1.0, height)
    col = tf.linspace(-1.0, 1.0, width)
    row_grid, col_grid = tf.meshgrid(row, col, indexing="ij")
    coords = tf.stack([row_grid, col_grid], axis=-1)
    coords = tf.reshape(coords, (1, height * width, 2))
    coords = tf.tile(coords, (batch_size, 1, 1))

    points = tf.reshape(x, (batch_size, height * width, channels))
    return tf.concat([points, coords], axis=-1)


def point_mlp_block(x, units, dropout_rate=0.2, name="point_mlp_block"):
    """Residual point-wise MLP block over unordered/fixed-grid point tokens."""
    shortcut = x
    if int(x.shape[-1]) != units:
        shortcut = Dense(units, name=f"{name}_shortcut")(shortcut)

    y = LayerNormalization(epsilon=1e-5, name=f"{name}_norm_1")(x)
    y = Dense(units, activation="elu", name=f"{name}_dense_1")(y)
    y = Dropout(dropout_rate, name=f"{name}_dropout_1")(y)
    y = Dense(units, name=f"{name}_dense_2")(y)
    x = Add(name=f"{name}_residual_1")([shortcut, y])

    shortcut = x
    y = LayerNormalization(epsilon=1e-5, name=f"{name}_norm_2")(x)
    value = Dense(units, activation="elu", name=f"{name}_value")(y)
    gate = Dense(units, activation="sigmoid", name=f"{name}_gate")(y)
    y = Multiply(name=f"{name}_gated_value")([value, gate])
    y = Dropout(dropout_rate, name=f"{name}_dropout_2")(y)
    x = Add(name=f"{name}_residual_2")([shortcut, y])
    return x


def attention_pool_points(x, name="point_attention_pool"):
    """Attention-weighted pooling over point tokens."""
    weights = Dense(1, name=f"{name}_score")(x)
    weights = tf.keras.layers.Softmax(axis=1, name=f"{name}_softmax")(weights)
    pooled = tf.keras.layers.Dot(axes=(1, 1), name=f"{name}_dot")([weights, x])
    return Flatten(name=f"{name}_flatten")(pooled)


def point_mlp_tiny(input_shape, dropout_rate=0.2):
    """Point-set MLP over 32x32 field cells with no custom serialization needs."""
    inputs = Input(shape=input_shape)
    n_points = input_shape[0] * input_shape[1]
    x = Reshape((n_points, input_shape[2]), name="point_tokens")(inputs)

    x = Dense(64, activation="elu", name="point_embed")(x)
    x = point_mlp_block(x, 64, dropout_rate=dropout_rate, name="point_block_1")
    x = point_mlp_block(x, 96, dropout_rate=dropout_rate, name="point_block_2")
    x = point_mlp_block(x, 128, dropout_rate=dropout_rate, name="point_block_3")
    x = LayerNormalization(epsilon=1e-5, name="point_final_norm")(x)

    mean_pool = GlobalAveragePooling1D(name="point_mean_pool")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="point_max_pool")(x)
    attn_pool = attention_pool_points(x, name="point_attention_pool")
    x = Concatenate(name="point_pool_concat")([mean_pool, max_pool, attn_pool])
    x = Dense(128, activation="elu", name="point_head_1")(x)
    x = Dropout(dropout_rate, name="point_head_dropout")(x)
    x = Dense(64, activation="elu", name="point_head_2")(x)
    model = Model(inputs, x)
    return model


def moga_block(x, filters, dropout_rate=0.2, name="moga_block"):
    """Multi-order gated aggregation block with H5-safe built-in Keras layers."""
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same", name=f"{name}_shortcut")(
            shortcut
        )

    y = LayerNormalization(epsilon=1e-5, name=f"{name}_spatial_norm")(x)
    value = Conv2D(filters, (1, 1), padding="same", name=f"{name}_value_proj")(y)
    gate_source = Conv2D(
        filters, (1, 1), padding="same", activation="elu", name=f"{name}_gate_source"
    )(y)

    branch_filters = max(filters // 4, 8)
    gate_1 = Conv2D(
        branch_filters,
        (1, 1),
        padding="same",
        activation="elu",
        name=f"{name}_gate_order_1",
    )(gate_source)
    gate_3 = Conv2D(
        branch_filters,
        (3, 3),
        padding="same",
        activation="elu",
        name=f"{name}_gate_order_3",
    )(gate_source)
    gate_5 = Conv2D(
        branch_filters,
        (5, 5),
        padding="same",
        activation="elu",
        name=f"{name}_gate_order_5",
    )(gate_source)
    gate_7 = Conv2D(
        branch_filters,
        (7, 7),
        padding="same",
        activation="elu",
        name=f"{name}_gate_order_7",
    )(gate_source)
    gate = Concatenate(name=f"{name}_multi_order_concat")(
        [gate_1, gate_3, gate_5, gate_7]
    )
    gate = Conv2D(
        filters, (1, 1), padding="same", activation="sigmoid", name=f"{name}_gate"
    )(gate)

    y = Multiply(name=f"{name}_gated_spatial_mix")([value, gate])
    y = Conv2D(filters, (1, 1), padding="same", name=f"{name}_spatial_out")(y)
    y = Dropout(dropout_rate, name=f"{name}_spatial_dropout")(y)
    x = Add(name=f"{name}_spatial_residual")([shortcut, y])

    shortcut = x
    y = LayerNormalization(epsilon=1e-5, name=f"{name}_channel_norm")(x)
    y = Conv2D(
        filters * 2,
        (1, 1),
        padding="same",
        activation="elu",
        name=f"{name}_channel_expand",
    )(y)
    y = Dropout(dropout_rate, name=f"{name}_channel_dropout_1")(y)
    y = Conv2D(filters, (1, 1), padding="same", name=f"{name}_channel_project")(y)
    y = Dropout(dropout_rate, name=f"{name}_channel_dropout_2")(y)
    x = Add(name=f"{name}_channel_residual")([shortcut, y])
    return x


def moganet_tiny(input_shape, dropout_rate=0.2):
    """Tiny MoGA-style regressor for 32x32 XCO2 fields."""
    inputs = Input(shape=input_shape)

    x = Conv2D(
        32, (3, 3), padding="same", activation="elu", name="moga_stem_conv_1"
    )(inputs)
    x = BatchNormalization(name="moga_stem_bn_1")(x)
    x = Conv2D(
        32, (3, 3), padding="same", activation="elu", name="moga_stem_conv_2"
    )(x)
    x = moga_block(x, 32, dropout_rate=dropout_rate / 2, name="moga_stage_1_block_1")
    x = MaxPooling2D((2, 2), name="moga_downsample_1")(x)

    x = Conv2D(
        64, (3, 3), padding="same", activation="elu", name="moga_stage_2_proj"
    )(x)
    x = BatchNormalization(name="moga_stage_2_bn")(x)
    x = moga_block(x, 64, dropout_rate=dropout_rate, name="moga_stage_2_block_1")
    x = moga_block(x, 64, dropout_rate=dropout_rate, name="moga_stage_2_block_2")
    x = MaxPooling2D((2, 2), name="moga_downsample_2")(x)

    x = Conv2D(
        96, (3, 3), padding="same", activation="elu", name="moga_stage_3_proj"
    )(x)
    x = BatchNormalization(name="moga_stage_3_bn")(x)
    x = moga_block(x, 96, dropout_rate=dropout_rate, name="moga_stage_3_block_1")
    x = moga_block(x, 96, dropout_rate=dropout_rate, name="moga_stage_3_block_2")

    x = Conv2D(
        128, (3, 3), padding="same", activation="elu", name="moga_stage_4_proj"
    )(x)
    x = BatchNormalization(name="moga_stage_4_bn")(x)
    x = moga_block(x, 128, dropout_rate=dropout_rate, name="moga_stage_4_block_1")

    x = LayerNormalization(epsilon=1e-5, name="moga_final_norm")(x)
    avg_pool = GlobalAveragePooling2D(name="moga_avg_pool")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D(name="moga_max_pool")(x)
    x = Concatenate(name="moga_pool_concat")([avg_pool, max_pool])
    x = Dense(128, activation="elu", name="moga_head_dense_1")(x)
    x = Dropout(dropout_rate, name="moga_head_dropout")(x)
    x = Dense(64, activation="elu", name="moga_head_dense_2")(x)
    model = Model(inputs, x)
    return model


def large_kernel_block(x, filters, dropout_rate=0.2, name="large_kernel_block"):
    """Bottleneck large-kernel residual block without depthwise/custom layers."""
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same", name=f"{name}_shortcut")(
            shortcut
        )

    y = LayerNormalization(epsilon=1e-5, name=f"{name}_norm_1")(x)
    mid_filters = max(filters // 4, 16)
    y = Conv2D(
        mid_filters,
        (1, 1),
        padding="same",
        activation="elu",
        name=f"{name}_reduce",
    )(y)

    branch_7 = Conv2D(
        mid_filters,
        (7, 7),
        padding="same",
        activation="elu",
        name=f"{name}_kernel_7",
    )(y)
    branch_11 = Conv2D(
        mid_filters,
        (11, 11),
        padding="same",
        activation="elu",
        name=f"{name}_kernel_11",
    )(y)
    y = Concatenate(name=f"{name}_multi_scale_concat")([branch_7, branch_11])
    y = Conv2D(filters, (1, 1), padding="same", name=f"{name}_expand")(y)
    y = Dropout(dropout_rate, name=f"{name}_spatial_dropout")(y)
    x = Add(name=f"{name}_spatial_residual")([shortcut, y])

    shortcut = x
    y = LayerNormalization(epsilon=1e-5, name=f"{name}_norm_2")(x)
    value = Conv2D(
        filters * 2,
        (1, 1),
        padding="same",
        activation="elu",
        name=f"{name}_ffn_value",
    )(y)
    gate = Conv2D(
        filters * 2,
        (1, 1),
        padding="same",
        activation="sigmoid",
        name=f"{name}_ffn_gate",
    )(y)
    y = Multiply(name=f"{name}_ffn_gated")([value, gate])
    y = Dropout(dropout_rate, name=f"{name}_ffn_dropout_1")(y)
    y = Conv2D(filters, (1, 1), padding="same", name=f"{name}_ffn_project")(y)
    y = Dropout(dropout_rate, name=f"{name}_ffn_dropout_2")(y)
    x = Add(name=f"{name}_ffn_residual")([shortcut, y])
    return x


def lk_cnn_tiny(input_shape, dropout_rate=0.2):
    """Large-kernel CNN for small XCO2 fields and plume-scale context."""
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="lk_stem_conv_1")(
        inputs
    )
    x = BatchNormalization(name="lk_stem_bn_1")(x)
    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="lk_stem_conv_2")(x)
    x = large_kernel_block(
        x, 32, dropout_rate=dropout_rate / 2, name="lk_stage_1_block_1"
    )
    x = MaxPooling2D((2, 2), name="lk_downsample_1")(x)

    x = Conv2D(64, (3, 3), padding="same", activation="elu", name="lk_stage_2_proj")(x)
    x = BatchNormalization(name="lk_stage_2_bn")(x)
    x = large_kernel_block(x, 64, dropout_rate=dropout_rate, name="lk_stage_2_block_1")
    x = MaxPooling2D((2, 2), name="lk_downsample_2")(x)

    x = Conv2D(96, (3, 3), padding="same", activation="elu", name="lk_stage_3_proj")(x)
    x = BatchNormalization(name="lk_stage_3_bn")(x)
    x = large_kernel_block(x, 96, dropout_rate=dropout_rate, name="lk_stage_3_block_1")
    x = large_kernel_block(x, 96, dropout_rate=dropout_rate, name="lk_stage_3_block_2")

    x = Conv2D(128, (3, 3), padding="same", activation="elu", name="lk_stage_4_proj")(
        x
    )
    x = BatchNormalization(name="lk_stage_4_bn")(x)
    x = large_kernel_block(x, 128, dropout_rate=dropout_rate, name="lk_stage_4_block_1")

    x = LayerNormalization(epsilon=1e-5, name="lk_final_norm")(x)
    avg_pool = GlobalAveragePooling2D(name="lk_avg_pool")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D(name="lk_max_pool")(x)
    x = Concatenate(name="lk_pool_concat")([avg_pool, max_pool])
    x = Dense(128, activation="elu", name="lk_head_dense_1")(x)
    x = Dropout(dropout_rate, name="lk_head_dropout")(x)
    x = Dense(64, activation="elu", name="lk_head_dense_2")(x)
    model = Model(inputs, x)
    return model


def star_block(x, filters, dropout_rate=0.2, name="star_block"):
    """StarNet-style multiplicative feature interaction block."""
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same", name=f"{name}_shortcut")(
            shortcut
        )

    y = LayerNormalization(epsilon=1e-5, name=f"{name}_norm_1")(x)
    y = Conv2D(filters, (3, 3), padding="same", activation="elu", name=f"{name}_local")(
        y
    )
    star_left = Conv2D(
        filters,
        (1, 1),
        padding="same",
        activation="elu",
        name=f"{name}_star_left",
    )(y)
    star_right = Conv2D(
        filters,
        (1, 1),
        padding="same",
        activation="elu",
        name=f"{name}_star_right",
    )(y)
    y = Multiply(name=f"{name}_star_multiply")([star_left, star_right])
    y = Conv2D(filters, (1, 1), padding="same", name=f"{name}_star_project")(y)
    y = Dropout(dropout_rate, name=f"{name}_star_dropout")(y)
    x = Add(name=f"{name}_star_residual")([shortcut, y])

    shortcut = x
    y = LayerNormalization(epsilon=1e-5, name=f"{name}_norm_2")(x)
    y = Conv2D(
        filters * 2,
        (1, 1),
        padding="same",
        activation="elu",
        name=f"{name}_ffn_expand",
    )(y)
    y = Dropout(dropout_rate, name=f"{name}_ffn_dropout_1")(y)
    y = Conv2D(filters, (1, 1), padding="same", name=f"{name}_ffn_project")(y)
    y = Dropout(dropout_rate, name=f"{name}_ffn_dropout_2")(y)
    x = Add(name=f"{name}_ffn_residual")([shortcut, y])
    return x


def starnet_tiny(input_shape, dropout_rate=0.2):
    """Tiny StarNet-style regressor using multiplicative feature interactions."""
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="star_stem_conv_1")(
        inputs
    )
    x = BatchNormalization(name="star_stem_bn_1")(x)
    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="star_stem_conv_2")(
        x
    )
    x = star_block(x, 32, dropout_rate=dropout_rate / 2, name="star_stage_1_block_1")
    x = MaxPooling2D((2, 2), name="star_downsample_1")(x)

    x = Conv2D(
        64, (3, 3), padding="same", activation="elu", name="star_stage_2_proj"
    )(x)
    x = BatchNormalization(name="star_stage_2_bn")(x)
    x = star_block(x, 64, dropout_rate=dropout_rate, name="star_stage_2_block_1")
    x = star_block(x, 64, dropout_rate=dropout_rate, name="star_stage_2_block_2")
    x = MaxPooling2D((2, 2), name="star_downsample_2")(x)

    x = Conv2D(
        96, (3, 3), padding="same", activation="elu", name="star_stage_3_proj"
    )(x)
    x = BatchNormalization(name="star_stage_3_bn")(x)
    x = star_block(x, 96, dropout_rate=dropout_rate, name="star_stage_3_block_1")
    x = star_block(x, 96, dropout_rate=dropout_rate, name="star_stage_3_block_2")

    x = Conv2D(
        128, (3, 3), padding="same", activation="elu", name="star_stage_4_proj"
    )(x)
    x = BatchNormalization(name="star_stage_4_bn")(x)
    x = star_block(x, 128, dropout_rate=dropout_rate, name="star_stage_4_block_1")

    x = LayerNormalization(epsilon=1e-5, name="star_final_norm")(x)
    avg_pool = GlobalAveragePooling2D(name="star_avg_pool")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D(name="star_max_pool")(x)
    x = Concatenate(name="star_pool_concat")([avg_pool, max_pool])
    x = Dense(128, activation="elu", name="star_head_dense_1")(x)
    x = Dropout(dropout_rate, name="star_head_dropout")(x)
    x = Dense(64, activation="elu", name="star_head_dense_2")(x)
    model = Model(inputs, x)
    return model


def bg_contrastive_branch(x, filters, dropout_rate=0.2, name="bg_contrastive_branch"):
    """Small residual encoder block for background or anomaly features."""
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same", name=f"{name}_shortcut")(
            shortcut
        )

    y = Conv2D(filters, (3, 3), padding="same", activation="elu", name=f"{name}_conv_1")(
        x
    )
    y = BatchNormalization(name=f"{name}_bn_1")(y)
    y = Dropout(dropout_rate, name=f"{name}_dropout_1")(y)
    y = Conv2D(filters, (3, 3), padding="same", activation="elu", name=f"{name}_conv_2")(
        y
    )
    y = BatchNormalization(name=f"{name}_bn_2")(y)
    x = Add(name=f"{name}_residual")([shortcut, y])
    return Activation("elu", name=f"{name}_activation")(x)


def bg_contrastive_cnn_tiny(input_shape, dropout_rate=0.2):
    """Background/anomaly dual-branch CNN for plume-sensitive regression.

    This minimal version only introduces the task-specific decomposition:
    multi-scale background branch + local anomaly branch + fused regression head.
    Contrastive losses can be added later if the architecture is promising.
    """
    inputs = Input(shape=input_shape)

    bg_3 = AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="bg_lowpass_pool_3x3",
    )(inputs)
    bg_5 = AveragePooling2D(
        pool_size=(5, 5),
        strides=(1, 1),
        padding="same",
        name="bg_lowpass_pool_5x5",
    )(inputs)
    bg_9 = AveragePooling2D(
        pool_size=(9, 9),
        strides=(1, 1),
        padding="same",
        name="bg_lowpass_pool_9x9",
    )(inputs)
    background_stack = Concatenate(name="bg_multiscale_stack")(
        [inputs, bg_3, bg_5, bg_9]
    )
    background = Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="elu",
        name="bg_learned_fusion_conv",
    )(background_stack)
    background = Conv2D(
        input_shape[-1],
        (1, 1),
        padding="same",
        name="bg_learned_background",
    )(background)
    anomaly = Subtract(name="bg_multiscale_anomaly")([inputs, background])
    anomaly_context = Concatenate(name="anom_context_stack")(
        [inputs, background, anomaly]
    )

    bg = Conv2D(24, (5, 5), padding="same", activation="elu", name="bg_stem_conv")(
        background
    )
    bg = BatchNormalization(name="bg_stem_bn")(bg)
    bg = bg_contrastive_branch(
        bg, 32, dropout_rate=dropout_rate / 2, name="bg_stage_1"
    )
    bg = MaxPooling2D((2, 2), name="bg_downsample_1")(bg)
    bg = bg_contrastive_branch(bg, 64, dropout_rate=dropout_rate, name="bg_stage_2")
    bg = MaxPooling2D((2, 2), name="bg_downsample_2")(bg)
    bg = bg_contrastive_branch(bg, 96, dropout_rate=dropout_rate, name="bg_stage_3")

    anom_value = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="elu",
        name="anom_context_value_conv",
    )(anomaly_context)
    anom_gate = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="sigmoid",
        name="anom_context_gate_conv",
    )(anomaly_context)
    anom = Multiply(name="anom_context_gated_value")([anom_value, anom_gate])
    anom = BatchNormalization(name="anom_stem_bn")(anom)
    anom = bg_contrastive_branch(
        anom, 32, dropout_rate=dropout_rate / 2, name="anom_stage_1"
    )
    anom = MaxPooling2D((2, 2), name="anom_downsample_1")(anom)
    anom = bg_contrastive_branch(
        anom, 64, dropout_rate=dropout_rate, name="anom_stage_2"
    )
    anom = MaxPooling2D((2, 2), name="anom_downsample_2")(anom)
    anom = bg_contrastive_branch(
        anom, 96, dropout_rate=dropout_rate, name="anom_stage_3"
    )

    cross = Multiply(name="bg_anom_cross_interaction")([bg, anom])
    diff = Subtract(name="bg_anom_feature_difference")([anom, bg])
    fused = Concatenate(name="bg_anom_feature_fusion")([bg, anom, cross, diff])
    fused = Conv2D(
        128, (1, 1), padding="same", activation="elu", name="fusion_projection"
    )(fused)
    fused = bg_contrastive_branch(
        fused, 128, dropout_rate=dropout_rate, name="fusion_stage_1"
    )

    avg_pool = GlobalAveragePooling2D(name="bg_contrastive_avg_pool")(fused)
    max_pool = tf.keras.layers.GlobalMaxPooling2D(name="bg_contrastive_max_pool")(fused)
    x = Concatenate(name="bg_contrastive_pool_concat")([avg_pool, max_pool])
    x = Dense(128, activation="elu", name="bg_contrastive_head_dense_1")(x)
    x = Dropout(dropout_rate, name="bg_contrastive_head_dropout")(x)
    x = Dense(64, activation="elu", name="bg_contrastive_head_dense_2")(x)
    model = Model(inputs, x)
    return model


def bg_plume_aux_cnn_tiny(input_shape, dropout_rate=0.2, classes=1):
    """Dual-branch inversion model with an auxiliary plume-map decoder."""
    inputs = Input(shape=input_shape)

    bg_3 = AveragePooling2D((3, 3), strides=(1, 1), padding="same", name="aux_bg_pool_3")(
        inputs
    )
    bg_5 = AveragePooling2D((5, 5), strides=(1, 1), padding="same", name="aux_bg_pool_5")(
        inputs
    )
    bg_9 = AveragePooling2D((9, 9), strides=(1, 1), padding="same", name="aux_bg_pool_9")(
        inputs
    )
    bg_stack = Concatenate(name="aux_bg_stack")([inputs, bg_3, bg_5, bg_9])
    background = Conv2D(
        16, (3, 3), padding="same", activation="elu", name="aux_bg_fusion_conv"
    )(bg_stack)
    background = Conv2D(input_shape[-1], (1, 1), padding="same", name="aux_background")(
        background
    )
    anomaly = Subtract(name="aux_anomaly")([inputs, background])
    anom_context = Concatenate(name="aux_anom_context")([inputs, background, anomaly])

    bg = Conv2D(24, (5, 5), padding="same", activation="elu", name="aux_bg_stem")(
        background
    )
    bg = BatchNormalization(name="aux_bg_stem_bn")(bg)
    bg = bg_contrastive_branch(bg, 32, dropout_rate / 2, name="aux_bg_stage_1")
    bg = MaxPooling2D((2, 2), name="aux_bg_down_1")(bg)
    bg = bg_contrastive_branch(bg, 64, dropout_rate, name="aux_bg_stage_2")
    bg = MaxPooling2D((2, 2), name="aux_bg_down_2")(bg)
    bg = bg_contrastive_branch(bg, 96, dropout_rate, name="aux_bg_stage_3")

    anom_value = Conv2D(
        32, (3, 3), padding="same", activation="elu", name="aux_anom_value"
    )(anom_context)
    anom_gate = Conv2D(
        32, (3, 3), padding="same", activation="sigmoid", name="aux_anom_gate"
    )(anom_context)
    anom = Multiply(name="aux_anom_gated_value")([anom_value, anom_gate])
    anom = BatchNormalization(name="aux_anom_stem_bn")(anom)
    anom = bg_contrastive_branch(anom, 32, dropout_rate / 2, name="aux_anom_stage_1")
    anom_down_1 = MaxPooling2D((2, 2), name="aux_anom_down_1")(anom)
    anom = bg_contrastive_branch(anom_down_1, 64, dropout_rate, name="aux_anom_stage_2")
    anom_down_2 = MaxPooling2D((2, 2), name="aux_anom_down_2")(anom)
    anom = bg_contrastive_branch(anom_down_2, 96, dropout_rate, name="aux_anom_stage_3")

    cross = Multiply(name="aux_bg_anom_cross")([bg, anom])
    diff = Subtract(name="aux_bg_anom_diff")([anom, bg])
    fused = Concatenate(name="aux_fusion")([bg, anom, cross, diff])
    fused = Conv2D(128, (1, 1), padding="same", activation="elu", name="aux_fuse_proj")(
        fused
    )
    fused = bg_contrastive_branch(fused, 128, dropout_rate, name="aux_fuse_stage")

    avg_pool = GlobalAveragePooling2D(name="aux_avg_pool")(fused)
    max_pool = tf.keras.layers.GlobalMaxPooling2D(name="aux_max_pool")(fused)
    reg = Concatenate(name="aux_reg_pool_concat")([avg_pool, max_pool])
    reg = Dense(128, activation="elu", name="aux_reg_dense_1")(reg)
    reg = Dropout(dropout_rate, name="aux_reg_dropout")(reg)
    reg = Dense(64, activation="elu", name="aux_reg_dense_2")(reg)
    emission = Dense(classes, name="emission")(reg)

    plume = Conv2D(64, (3, 3), padding="same", activation="elu", name="aux_dec_conv_1")(
        fused
    )
    plume = Conv2DTranspose(
        64,
        (2, 2),
        strides=(2, 2),
        padding="same",
        activation="elu",
        name="aux_dec_up_1",
    )(plume)
    plume = Concatenate(name="aux_dec_skip_1")([plume, anom_down_1])
    plume = Conv2D(48, (3, 3), padding="same", activation="elu", name="aux_dec_conv_2")(
        plume
    )
    plume = Conv2DTranspose(
        48,
        (2, 2),
        strides=(2, 2),
        padding="same",
        activation="elu",
        name="aux_dec_up_2",
    )(plume)
    plume = Concatenate(name="aux_dec_skip_2")([plume, anom_context])
    plume = Conv2D(32, (3, 3), padding="same", activation="elu", name="aux_dec_conv_3")(
        plume
    )
    plume_aux = Conv2D(
        1, (1, 1), padding="same", activation="sigmoid", name="plume_aux"
    )(plume)

    return Model(inputs, {"emission": emission, "plume_aux": plume_aux})


def scale_aware_cnn_tiny(input_shape, dropout_rate=0.2, classes=1, n_bins=4):
    """Essential CNN with an auxiliary emission-scale classification head."""
    inputs = Input(shape=input_shape)
    features = essential_regressor(input_shape, dropout_rate)(inputs)

    reg = Dense(64, activation="elu", name="scale_reg_dense")(features)
    reg = Dropout(dropout_rate, name="scale_reg_dropout")(reg)
    emission = Dense(classes, name="emission")(reg)

    bin_head = Dense(64, activation="elu", name="scale_bin_dense")(features)
    bin_head = Dropout(dropout_rate, name="scale_bin_dropout")(bin_head)
    emission_bin = Dense(n_bins, activation="softmax", name="emission_bin")(bin_head)

    return Model(inputs, {"emission": emission, "emission_bin": emission_bin})


def scale_decomposed_cnn_tiny(input_shape, dropout_rate=0.2, classes=1, n_bins=4):
    """
    Scale-decomposed regressor.

    The model first predicts a coarse emission bin, then predicts a bounded
    within-bin residual. The final emission is differentiably composed from
    the soft bin center and residual, reducing collapse toward mid emissions.
    """
    inputs = Input(shape=input_shape)
    features = essential_regressor(input_shape, dropout_rate)(inputs)

    shared = Dense(96, activation="elu", name="sd_shared_dense")(features)
    shared = Dropout(dropout_rate, name="sd_shared_dropout")(shared)

    bin_head = Dense(64, activation="elu", name="sd_bin_dense")(shared)
    emission_bin = Dense(
        n_bins,
        activation="softmax",
        name="emission_bin",
    )(bin_head)

    residual_head = Dense(64, activation="elu", name="sd_residual_dense")(shared)
    residual = Dense(classes, name="sd_residual_raw")(residual_head)
    residual = Activation("tanh", name="sd_residual_tanh")(residual)

    center_layer = Dense(
        classes,
        use_bias=False,
        trainable=False,
        name="sd_bin_center",
    )
    center = center_layer(emission_bin)
    residual_scale_layer = Dense(
        classes,
        use_bias=False,
        trainable=False,
        name="sd_residual_scale",
    )
    residual_scaled = residual_scale_layer(residual)
    emission = Add(name="emission")([center, residual_scaled])

    model = Model(inputs, {"emission": emission, "emission_bin": emission_bin})
    center_layer.set_weights([np.array([[4.0], [13.0], [24.0], [42.0]], dtype=np.float32)])
    residual_scale_layer.set_weights([np.array([[12.0]], dtype=np.float32)])
    return model


## TRANSFORMER

def spectral_residual_amp_phase_transform(x):
    """Return log-amplitude and phase maps from local XCO2 residuals."""
    xco2 = x[..., 0]
    xco2_4d = x[..., 0:1]
    local_bg = tf.nn.avg_pool2d(
        xco2_4d,
        ksize=9,
        strides=1,
        padding="SAME",
    )
    residual = xco2_4d - local_bg
    residual = residual[..., 0]
    residual = residual - tf.reduce_mean(residual, axis=[1, 2], keepdims=True)
    fft = tf.signal.fft2d(tf.cast(residual, tf.complex64))
    amplitude = tf.math.log1p(tf.abs(fft))
    phase = tf.math.angle(fft)
    return tf.stack(
        [amplitude, tf.math.sin(phase), tf.math.cos(phase)],
        axis=-1,
    )


def spectral_amp_phase_cnn_tiny(input_shape, dropout_rate=0.2):
    """
    Spatial CNN plus Fourier amplitude/phase branch on local XCO2 residuals.

    The spectral branch first subtracts a local mean background from XCO2, then
    extracts amplitude/phase features from the residual field. This reduces the
    chance that broad real-background gradients dominate the spectral features.
    """
    inputs = Input(shape=input_shape)

    sp = Conv2D(32, (3, 3), padding="same", activation="elu", name="sap_sp_conv_1")(
        inputs
    )
    sp = BatchNormalization(name="sap_sp_bn_1")(sp)
    sp = Conv2D(32, (3, 3), padding="same", activation="elu", name="sap_sp_conv_2")(
        sp
    )
    sp = MaxPooling2D((2, 2), name="sap_sp_pool_1")(sp)
    sp = Dropout(dropout_rate / 2, name="sap_sp_dropout_1")(sp)

    sp = Conv2D(64, (3, 3), padding="same", activation="elu", name="sap_sp_conv_3")(
        sp
    )
    sp = BatchNormalization(name="sap_sp_bn_2")(sp)
    sp = Conv2D(64, (3, 3), padding="same", activation="elu", name="sap_sp_conv_4")(
        sp
    )
    sp = MaxPooling2D((2, 2), name="sap_sp_pool_2")(sp)
    sp = Dropout(dropout_rate, name="sap_sp_dropout_2")(sp)
    sp = GlobalAveragePooling2D(name="sap_sp_pool")(sp)

    spec = Lambda(
        spectral_residual_amp_phase_transform,
        name="sap_residual_fft_amp_phase",
    )(inputs)
    spec = Conv2D(
        24, (1, 1), padding="same", activation="elu", name="sap_spec_project"
    )(spec)
    spec = BatchNormalization(name="sap_spec_bn_1")(spec)
    spec = Conv2D(
        32, (3, 3), padding="same", activation="elu", name="sap_spec_conv_1"
    )(spec)
    spec = MaxPooling2D((2, 2), name="sap_spec_pool_1")(spec)
    spec = Dropout(dropout_rate / 2, name="sap_spec_dropout_1")(spec)
    spec = Conv2D(
        64, (3, 3), padding="same", activation="elu", name="sap_spec_conv_2"
    )(spec)
    spec = MaxPooling2D((2, 2), name="sap_spec_pool_2")(spec)
    spec = GlobalAveragePooling2D(name="sap_spec_pool")(spec)

    x = Concatenate(name="sap_fuse_spatial_spectral")([sp, spec])
    x = Dense(128, activation="elu", name="sap_fuse_dense_1")(x)
    x = Dropout(dropout_rate, name="sap_fuse_dropout")(x)
    x = Dense(64, activation="elu", name="sap_fuse_dense_2")(x)
    return Model(inputs, x, name="spectral_amp_phase_cnn_tiny_core")


def multi_scale_xco2_residuals(x):
    """Build local-background residual maps from the XCO2 channel."""
    xco2 = x[..., 0:1]
    residuals = []
    for ksize in (5, 9, 15):
        local_bg = tf.nn.avg_pool2d(
            xco2,
            ksize=ksize,
            strides=1,
            padding="SAME",
        )
        residuals.append(xco2 - local_bg)
    return tf.concat(residuals, axis=-1)


def uv_channels(x):
    """Return u/v wind channels from an XCO2/u/v input tensor."""
    return x[..., 1:3]


def wind_speed_channel(uv):
    """Return wind-speed magnitude as a single image channel."""
    return tf.sqrt(tf.reduce_sum(tf.square(uv), axis=-1, keepdims=True) + 1e-6)


def spatial_center_maps(x):
    """Return normalized x/y coordinates and a center prior map."""
    shape = tf.shape(x)
    batch = shape[0]
    height = shape[1]
    width = shape[2]
    yy = tf.linspace(-1.0, 1.0, height)
    xx = tf.linspace(-1.0, 1.0, width)
    grid_y, grid_x = tf.meshgrid(yy, xx, indexing="ij")
    center = tf.exp(-4.0 * (tf.square(grid_x) + tf.square(grid_y)))
    maps = tf.stack([grid_x, grid_y, center], axis=-1)
    maps = tf.expand_dims(maps, axis=0)
    return tf.tile(maps, [batch, 1, 1, 1])


def sum_over_observations(x):
    """Sum K observation embeddings after attention weighting."""
    return tf.reduce_sum(x, axis=1)


def hpr_branch_encoder(x, filters, dropout_rate=0.2, name="hpr_branch"):
    """Small CNN branch encoder used by the hierarchical plume reasoner."""
    x = Conv2D(
        filters,
        (3, 3),
        padding="same",
        activation="elu",
        name=f"{name}_conv_1",
    )(x)
    x = BatchNormalization(name=f"{name}_bn_1")(x)
    x = Conv2D(
        filters,
        (3, 3),
        padding="same",
        activation="elu",
        name=f"{name}_conv_2",
    )(x)
    x = MaxPooling2D((2, 2), name=f"{name}_pool_1")(x)
    x = Dropout(dropout_rate / 2, name=f"{name}_dropout_1")(x)
    x = Conv2D(
        filters * 2,
        (3, 3),
        padding="same",
        activation="elu",
        name=f"{name}_conv_3",
    )(x)
    x = MaxPooling2D((2, 2), name=f"{name}_pool_2")(x)
    x = Conv2D(
        filters * 2,
        (3, 3),
        padding="same",
        activation="elu",
        name=f"{name}_conv_4",
    )(x)
    return GlobalAveragePooling2D(name=f"{name}_gap")(x)


def multi_scale_residual_cnn_tiny(input_shape, dropout_rate=0.2):
    """
    CNN using original XCO2/u/v plus multi-scale XCO2 residual maps.

    The residual maps emphasize plume-like anomalies after subtracting local
    background estimates at several spatial scales, avoiding FFT/complex ops.
    """
    inputs = Input(shape=input_shape)
    residuals = Lambda(
        multi_scale_xco2_residuals,
        name="msr_xco2_residuals_5_9_15",
    )(inputs)
    x = Concatenate(name="msr_input_plus_residuals")([inputs, residuals])

    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="msr_stem_conv_1")(
        x
    )
    x = BatchNormalization(name="msr_stem_bn_1")(x)
    x = Conv2D(32, (3, 3), padding="same", activation="elu", name="msr_stem_conv_2")(
        x
    )
    x = MaxPooling2D((2, 2), name="msr_pool_1")(x)
    x = Dropout(dropout_rate / 2, name="msr_dropout_1")(x)

    x = Conv2D(64, (3, 3), padding="same", activation="elu", name="msr_stage2_conv_1")(
        x
    )
    x = BatchNormalization(name="msr_stage2_bn")(x)
    skip = Conv2D(64, (1, 1), padding="same", name="msr_stage2_skip")(x)
    y = Conv2D(64, (3, 3), padding="same", activation="elu", name="msr_stage2_conv_2")(
        x
    )
    y = Conv2D(64, (3, 3), padding="same", name="msr_stage2_conv_3")(y)
    x = Add(name="msr_stage2_add")([skip, y])
    x = Activation("elu", name="msr_stage2_act")(x)
    x = MaxPooling2D((2, 2), name="msr_pool_2")(x)
    x = Dropout(dropout_rate, name="msr_dropout_2")(x)

    x = Conv2D(96, (3, 3), padding="same", activation="elu", name="msr_stage3_conv_1")(
        x
    )
    x = BatchNormalization(name="msr_stage3_bn")(x)
    x = Conv2D(128, (3, 3), padding="same", activation="elu", name="msr_stage3_conv_2")(
        x
    )
    x = GlobalAveragePooling2D(name="msr_global_pool")(x)

    x = Dense(96, activation="elu", name="msr_head_dense_1")(x)
    x = Dropout(dropout_rate, name="msr_head_dropout")(x)
    x = Dense(64, activation="elu", name="msr_head_dense_2")(x)
    return Model(inputs, x, name="multi_scale_residual_cnn_tiny_core")


def wind_guided_residual_cnn_tiny(input_shape, dropout_rate=0.2):
    """
    Wind-guided residual CNN for plume inversion.

    XCO2 residual maps emphasize plume-like anomalies, while the u/v channels
    produce a spatial gate that modulates residual features according to wind
    context. This targets wind-aligned plume structures instead of background
    bright spots.
    """
    inputs = Input(shape=input_shape)

    residuals = Lambda(
        multi_scale_xco2_residuals,
        name="wgr_xco2_residuals_5_9_15",
    )(inputs)
    wind = Lambda(lambda t: t[..., 1:3], name="wgr_uv_channels")(inputs)
    wind_speed = Lambda(
        lambda t: tf.sqrt(tf.reduce_sum(tf.square(t), axis=-1, keepdims=True) + 1e-6),
        name="wgr_wind_speed",
    )(wind)
    wind_context = Concatenate(name="wgr_wind_context")([wind, wind_speed])

    res = Conv2D(32, (3, 3), padding="same", activation="elu", name="wgr_res_conv_1")(
        residuals
    )
    res = BatchNormalization(name="wgr_res_bn_1")(res)
    res = Conv2D(48, (3, 3), padding="same", activation="elu", name="wgr_res_conv_2")(
        res
    )

    gate = Conv2D(24, (3, 3), padding="same", activation="elu", name="wgr_gate_conv_1")(
        wind_context
    )
    gate = Conv2D(48, (1, 1), padding="same", activation="sigmoid", name="wgr_gate")(
        gate
    )
    guided = Multiply(name="wgr_guided_residual")([res, gate])

    raw = Conv2D(24, (3, 3), padding="same", activation="elu", name="wgr_raw_conv_1")(
        inputs
    )
    x = Concatenate(name="wgr_fuse_raw_guided")([raw, guided, wind_context])

    x = Conv2D(64, (3, 3), padding="same", activation="elu", name="wgr_stage1_conv")(
        x
    )
    x = BatchNormalization(name="wgr_stage1_bn")(x)
    x = MaxPooling2D((2, 2), name="wgr_pool_1")(x)
    x = Dropout(dropout_rate / 2, name="wgr_dropout_1")(x)

    x = Conv2D(96, (3, 3), padding="same", activation="elu", name="wgr_stage2_conv_1")(
        x
    )
    skip = Conv2D(96, (1, 1), padding="same", name="wgr_stage2_skip")(x)
    y = Conv2D(96, (3, 3), padding="same", activation="elu", name="wgr_stage2_conv_2")(
        x
    )
    y = Conv2D(96, (3, 3), padding="same", name="wgr_stage2_conv_3")(y)
    x = Add(name="wgr_stage2_add")([skip, y])
    x = Activation("elu", name="wgr_stage2_act")(x)
    x = MaxPooling2D((2, 2), name="wgr_pool_2")(x)
    x = Dropout(dropout_rate, name="wgr_dropout_2")(x)

    x = Conv2D(128, (3, 3), padding="same", activation="elu", name="wgr_stage3_conv")(
        x
    )
    x = GlobalAveragePooling2D(name="wgr_global_pool")(x)
    x = Dense(96, activation="elu", name="wgr_head_dense_1")(x)
    x = Dropout(dropout_rate, name="wgr_head_dropout")(x)
    x = Dense(64, activation="elu", name="wgr_head_dense_2")(x)
    return Model(inputs, x, name="wind_guided_residual_cnn_tiny_core")


def hierarchical_plume_reasoner_tiny(input_shape, dropout_rate=0.2):
    """
    Minimal hierarchical plume-reasoning CNN for real-background plume inversion.

    The model separates three evidence levels before fusion:
    1) background/residual evidence from multi-scale XCO2 residual maps;
    2) plume-morphology evidence from positive residuals and center prior;
    3) physical-context evidence from u/v wind, wind speed, and coordinates.

    It intentionally avoids FFT, Transformer, and auxiliary losses for a
    stop-loss single-seed test against the current essential CNN baseline.
    """
    inputs = Input(shape=input_shape)

    residuals = Lambda(
        multi_scale_xco2_residuals,
        name="hpr_xco2_residuals_5_9_15",
    )(inputs)
    positive_residuals = Lambda(
        lambda t: tf.nn.relu(t),
        name="hpr_positive_residuals",
    )(residuals)
    coord_maps = Lambda(spatial_center_maps, name="hpr_spatial_center_maps")(inputs)

    if int(input_shape[-1]) >= 3:
        uv = Lambda(uv_channels, name="hpr_uv_channels")(inputs)
        speed = Lambda(wind_speed_channel, name="hpr_wind_speed")(uv)
        wind_context = Concatenate(name="hpr_wind_context")([uv, speed, coord_maps])
    else:
        wind_context = coord_maps

    low_input = Concatenate(name="hpr_low_input")([inputs, residuals])
    low_feat = hpr_branch_encoder(
        low_input,
        filters=32,
        dropout_rate=dropout_rate,
        name="hpr_low_residual_branch",
    )

    mid_input = Concatenate(name="hpr_mid_input")(
        [positive_residuals, residuals, coord_maps]
    )
    mid_feat = hpr_branch_encoder(
        mid_input,
        filters=32,
        dropout_rate=dropout_rate,
        name="hpr_mid_morphology_branch",
    )

    phys_feat = hpr_branch_encoder(
        wind_context,
        filters=24,
        dropout_rate=dropout_rate,
        name="hpr_physics_branch",
    )

    raw_feat = essential_regressor(input_shape, dropout_rate)(inputs)
    raw_feat = Dense(64, activation="elu", name="hpr_raw_essential_projection")(
        raw_feat
    )

    branch_stack = Concatenate(name="hpr_branch_feature_stack")(
        [raw_feat, low_feat, mid_feat, phys_feat]
    )
    branch_stack = Dense(128, activation="elu", name="hpr_fusion_pre_dense")(
        branch_stack
    )
    branch_stack = Dropout(dropout_rate, name="hpr_fusion_dropout")(branch_stack)

    gate = Dense(4, activation="sigmoid", name="hpr_branch_gate")(branch_stack)
    gate_raw = Lambda(lambda t: t[:, 0:1], name="hpr_gate_raw")(gate)
    gate_low = Lambda(lambda t: t[:, 1:2], name="hpr_gate_low")(gate)
    gate_mid = Lambda(lambda t: t[:, 2:3], name="hpr_gate_mid")(gate)
    gate_phys = Lambda(lambda t: t[:, 3:4], name="hpr_gate_phys")(gate)

    gated_raw = Multiply(name="hpr_gated_raw")([raw_feat, gate_raw])
    gated_low = Multiply(name="hpr_gated_low")([low_feat, gate_low])
    gated_mid = Multiply(name="hpr_gated_mid")([mid_feat, gate_mid])
    gated_phys = Multiply(name="hpr_gated_phys")([phys_feat, gate_phys])

    x = Concatenate(name="hpr_gated_feature_fusion")(
        [gated_raw, gated_low, gated_mid, gated_phys]
    )
    x = Dense(128, activation="elu", name="hpr_head_dense_1")(x)
    x = Dropout(dropout_rate, name="hpr_head_dropout")(x)
    x = Dense(64, activation="elu", name="hpr_head_dense_2")(x)
    return Model(inputs, x, name="hierarchical_plume_reasoner_tiny_core")


def active_plume_region_cnn_tiny(input_shape, dropout_rate=0.2):
    """
    Active plume-region CNN for real-background mixed inversion.

    The model first computes multi-scale XCO2 residual maps, uses residual and
    wind context to infer a soft plume-region mask, and then combines full-image
    CNN features with mask-weighted plume-region features for emission
    regression. The mask is not supervised; it acts as an inductive bias to
    reduce sensitivity to background bright spots and missing-pixel boundaries.
    """
    inputs = Input(shape=input_shape)

    residuals = Lambda(
        multi_scale_xco2_residuals,
        name="apr_xco2_residuals_5_9_15",
    )(inputs)

    if int(input_shape[-1]) >= 3:
        uv = Lambda(uv_channels, name="apr_uv_channels")(inputs)
        speed = Lambda(wind_speed_channel, name="apr_wind_speed")(uv)
        wind_context = Concatenate(name="apr_wind_context")([uv, speed])
    else:
        wind_context = residuals

    full = Concatenate(name="apr_full_input")([inputs, residuals])
    full = Conv2D(32, (3, 3), padding="same", activation="elu", name="apr_full_conv_1")(
        full
    )
    full = BatchNormalization(name="apr_full_bn_1")(full)
    full = Conv2D(48, (3, 3), padding="same", activation="elu", name="apr_full_conv_2")(
        full
    )
    full = MaxPooling2D((2, 2), name="apr_full_pool_1")(full)
    full = Dropout(dropout_rate / 2, name="apr_full_dropout_1")(full)
    full = Conv2D(96, (3, 3), padding="same", activation="elu", name="apr_full_conv_3")(
        full
    )
    full = MaxPooling2D((2, 2), name="apr_full_pool_2")(full)
    full_features = GlobalAveragePooling2D(name="apr_full_global_pool")(full)

    plume_features = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="elu",
        name="apr_plume_res_conv_1",
    )(residuals)
    plume_features = BatchNormalization(name="apr_plume_res_bn_1")(plume_features)
    plume_features = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="elu",
        name="apr_plume_res_conv_2",
    )(plume_features)

    mask_input = Concatenate(name="apr_mask_input")([residuals, wind_context])
    mask = Conv2D(24, (3, 3), padding="same", activation="elu", name="apr_mask_conv_1")(
        mask_input
    )
    mask = Conv2D(16, (3, 3), padding="same", activation="elu", name="apr_mask_conv_2")(
        mask
    )
    mask = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name="apr_soft_mask")(
        mask
    )

    gated_plume = Multiply(name="apr_masked_plume_features")([plume_features, mask])
    gated_plume = Conv2D(
        96,
        (3, 3),
        padding="same",
        activation="elu",
        name="apr_masked_plume_conv",
    )(gated_plume)
    plume_avg = GlobalAveragePooling2D(name="apr_plume_avg_pool")(gated_plume)
    plume_max = tf.keras.layers.GlobalMaxPooling2D(name="apr_plume_max_pool")(
        gated_plume
    )
    mask_summary = GlobalAveragePooling2D(name="apr_mask_mean")(mask)

    wind_features = GlobalAveragePooling2D(name="apr_wind_global_pool")(wind_context)
    wind_features = Dense(24, activation="elu", name="apr_wind_dense")(wind_features)

    x = Concatenate(name="apr_feature_fusion")(
        [full_features, plume_avg, plume_max, mask_summary, wind_features]
    )
    x = Dense(128, activation="elu", name="apr_head_dense_1")(x)
    x = Dropout(dropout_rate, name="apr_head_dropout")(x)
    x = Dense(64, activation="elu", name="apr_head_dense_2")(x)
    return Model(inputs, x, name="active_plume_region_cnn_tiny_core")


def supervised_plume_attention_cnn_tiny(input_shape, dropout_rate=0.2, classes=1):
    """
    Plume-attention inversion model with weak plume-mask supervision.

    The internal attention mask is predicted from multi-scale XCO2 residuals
    and wind context, then used to pool plume-region features for emission
    regression. The same mask is exposed as the ``plume_aux`` output so it can
    be weakly supervised by the known simulated plume map in mixed samples.
    """
    inputs = Input(shape=input_shape)

    residuals = Lambda(
        multi_scale_xco2_residuals,
        name="spa_xco2_residuals_5_9_15",
    )(inputs)

    if int(input_shape[-1]) >= 3:
        uv = Lambda(uv_channels, name="spa_uv_channels")(inputs)
        speed = Lambda(wind_speed_channel, name="spa_wind_speed")(uv)
        wind_context = Concatenate(name="spa_wind_context")([uv, speed])
    else:
        wind_context = residuals

    mask_input = Concatenate(name="spa_mask_input")([residuals, wind_context])
    mask = Conv2D(24, (3, 3), padding="same", activation="elu", name="spa_mask_conv_1")(
        mask_input
    )
    mask = BatchNormalization(name="spa_mask_bn_1")(mask)
    mask = Conv2D(16, (3, 3), padding="same", activation="elu", name="spa_mask_conv_2")(
        mask
    )
    plume_aux = Conv2D(
        1,
        (1, 1),
        padding="same",
        activation="sigmoid",
        name="plume_aux",
    )(mask)

    full = Concatenate(name="spa_full_input")([inputs, residuals])
    full = Conv2D(32, (3, 3), padding="same", activation="elu", name="spa_full_conv_1")(
        full
    )
    full = BatchNormalization(name="spa_full_bn_1")(full)
    full = Conv2D(48, (3, 3), padding="same", activation="elu", name="spa_full_conv_2")(
        full
    )
    full = MaxPooling2D((2, 2), name="spa_full_pool_1")(full)
    full = Dropout(dropout_rate / 2, name="spa_full_dropout_1")(full)
    full = Conv2D(96, (3, 3), padding="same", activation="elu", name="spa_full_conv_3")(
        full
    )
    full = MaxPooling2D((2, 2), name="spa_full_pool_2")(full)
    full_features = GlobalAveragePooling2D(name="spa_full_global_pool")(full)

    plume_features = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="elu",
        name="spa_plume_conv_1",
    )(residuals)
    plume_features = BatchNormalization(name="spa_plume_bn_1")(plume_features)
    plume_features = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="elu",
        name="spa_plume_conv_2",
    )(plume_features)
    plume_features = Multiply(name="spa_attention_weighted_plume")(
        [plume_features, plume_aux]
    )
    plume_features = Conv2D(
        96,
        (3, 3),
        padding="same",
        activation="elu",
        name="spa_plume_weighted_conv",
    )(plume_features)
    plume_avg = GlobalAveragePooling2D(name="spa_plume_avg_pool")(plume_features)
    plume_max = tf.keras.layers.GlobalMaxPooling2D(name="spa_plume_max_pool")(
        plume_features
    )

    mask_summary = GlobalAveragePooling2D(name="spa_mask_mean")(plume_aux)
    wind_features = GlobalAveragePooling2D(name="spa_wind_global_pool")(wind_context)
    wind_features = Dense(24, activation="elu", name="spa_wind_dense")(wind_features)

    x = Concatenate(name="spa_feature_fusion")(
        [full_features, plume_avg, plume_max, mask_summary, wind_features]
    )
    x = Dense(128, activation="elu", name="spa_head_dense_1")(x)
    x = Dropout(dropout_rate, name="spa_head_dropout")(x)
    x = Dense(64, activation="elu", name="spa_head_dense_2")(x)
    emission = Dense(classes, name="emission")(x)

    return Model(inputs, {"emission": emission, "plume_aux": plume_aux})


def attention_feature_aux_v2_cnn_tiny(input_shape, dropout_rate=0.2, classes=1):
    """
    Essential-CNN regressor with a weakly supervised attention feature branch.

    Unlike supervised_plume_attention_cnn_tiny, the attention mask does not gate
    the main regression pathway. The full-image essential CNN remains the
    primary regressor, while the plume-attention branch contributes auxiliary
    pooled features and is weakly supervised through the known simulated plume
    map. This reduces the risk that an imperfect mask suppresses useful global
    emission-scale information.
    """
    inputs = Input(shape=input_shape)

    full_features = essential_regressor(input_shape, dropout_rate)(inputs)

    residuals = Lambda(
        multi_scale_xco2_residuals,
        name="afav2_xco2_residuals_5_9_15",
    )(inputs)

    if int(input_shape[-1]) >= 3:
        uv = Lambda(uv_channels, name="afav2_uv_channels")(inputs)
        speed = Lambda(wind_speed_channel, name="afav2_wind_speed")(uv)
        wind_context = Concatenate(name="afav2_wind_context")([uv, speed])
    else:
        wind_context = residuals

    mask_input = Concatenate(name="afav2_mask_input")([residuals, wind_context])
    mask = Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="elu",
        name="afav2_mask_conv_1",
    )(mask_input)
    mask = BatchNormalization(name="afav2_mask_bn_1")(mask)
    mask = Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="elu",
        name="afav2_mask_conv_2",
    )(mask)
    plume_aux = Conv2D(
        1,
        (1, 1),
        padding="same",
        activation="sigmoid",
        name="plume_aux",
    )(mask)

    plume_features = Conv2D(
        24,
        (3, 3),
        padding="same",
        activation="elu",
        name="afav2_plume_conv_1",
    )(residuals)
    plume_features = BatchNormalization(name="afav2_plume_bn_1")(plume_features)
    plume_features = Conv2D(
        48,
        (3, 3),
        padding="same",
        activation="elu",
        name="afav2_plume_conv_2",
    )(plume_features)

    weighted_plume = Multiply(name="afav2_attention_weighted_plume")(
        [plume_features, plume_aux]
    )
    plume_avg = GlobalAveragePooling2D(name="afav2_plume_avg_pool")(weighted_plume)
    plume_max = tf.keras.layers.GlobalMaxPooling2D(name="afav2_plume_max_pool")(
        weighted_plume
    )
    mask_summary = GlobalAveragePooling2D(name="afav2_mask_mean")(plume_aux)
    wind_features = GlobalAveragePooling2D(name="afav2_wind_global_pool")(wind_context)
    wind_features = Dense(16, activation="elu", name="afav2_wind_dense")(wind_features)

    x = Concatenate(name="afav2_feature_fusion")(
        [full_features, plume_avg, plume_max, mask_summary, wind_features]
    )
    x = Dense(96, activation="elu", name="afav2_head_dense_1")(x)
    x = Dropout(dropout_rate, name="afav2_head_dropout")(x)
    x = Dense(64, activation="elu", name="afav2_head_dense_2")(x)
    emission = Dense(classes, name="emission")(x)

    return Model(inputs, {"emission": emission, "plume_aux": plume_aux})


def observation_set_encoder_tiny(
    input_shape,
    dropout_rate=0.2,
    classes=1,
    set_size=4,
):
    """
    Tiny observation-set encoder for plant-level emission inversion.

    Each OCO-3-like scene is encoded by a shared essential CNN. Scene embeddings
    are then aggregated with learned attention pooling, allowing the model to
    learn which observations in a same-emission set are most informative instead
    of treating every 32x32 scene independently.
    """
    scene_shape = tuple(input_shape)
    inputs = Input(shape=(set_size,) + scene_shape, name="observation_set_input")

    scene_encoder = essential_regressor(scene_shape, dropout_rate)
    embeddings = tf.keras.layers.TimeDistributed(
        scene_encoder,
        name="ose_scene_encoder",
    )(inputs)

    hidden = Dense(96, activation="elu", name="ose_scene_dense")(embeddings)
    hidden = Dropout(dropout_rate, name="ose_scene_dropout")(hidden)
    attn_logits = Dense(1, name="ose_attention_logits")(hidden)
    attn_weights = Activation("softmax", name="ose_attention_weights")(attn_logits)
    weighted_embeddings = Multiply(name="ose_attention_weighted_embeddings")(
        [embeddings, attn_weights]
    )
    set_embedding = Lambda(
        sum_over_observations,
        name="ose_attention_pool",
    )(weighted_embeddings)

    mean_embedding = GlobalAveragePooling1D(name="ose_mean_pool")(embeddings)
    x = Concatenate(name="ose_fuse_attention_mean")([set_embedding, mean_embedding])
    x = Dense(128, activation="elu", name="ose_head_dense_1")(x)
    x = Dropout(dropout_rate, name="ose_head_dropout")(x)
    x = Dense(64, activation="elu", name="ose_head_dense_2")(x)
    outputs = Dense(classes, name="regressor")(x)
    return Model(inputs, outputs, name="observation_set_encoder_tiny")


from tensorflow import keras
from tensorflow.keras import layers

positional_emb = True
conv_layers = 2

transformer_layers = 2
stochastic_depth_rate = 0.1


class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers=conv_layers,
        num_output_channels=[64, 64],
        positional_emb=positional_emb,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # This is our tokenizer.
        self.conv_model = keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(
                layers.Conv2D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(layers.ZeroPadding2D(padding))
            self.conv_model.add(
                layers.MaxPooling2D(pooling_kernel_size, pooling_stride, "same")
            )

        self.positional_emb = positional_emb

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = tf.reshape(
            outputs,
            (
                -1,
                tf.shape(outputs)[1] * tf.shape(outputs)[2],
                tf.shape(outputs)[-1],
            ),
        )
        return reshaped


class PositionEmbedding(keras.layers.Layer):
    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError("`sequence_length` must be an Integer, received `None`.")
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, start_index=0):
        inputs_shape = tf.shape(inputs)
        feature_length = inputs_shape[-1]
        sequence_length = inputs_shape[-2]

        position_embeddings = tf.convert_to_tensor(self.position_embeddings)
        position_embeddings = tf.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return tf.broadcast_to(position_embeddings, inputs_shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class SequencePooling(layers.Layer):
    def __init__(self):
        super().__init__()
        self.attention = layers.Dense(1)

    def call(self, x):
        attention_weights = tf.nn.softmax(self.attention(x), axis=1)
        attention_weights = tf.transpose(attention_weights, perm=(0, 2, 1))
        weighted_representation = tf.matmul(attention_weights, x)
        return tf.squeeze(weighted_representation, -2)


class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop
        self.seed_generator = tf.random.set_seed(1337)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (np.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(
                shape, 0, 1, seed=self.seed_generator
            )
            random_tensor = tf.math.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation="elu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


projection_dim = 64

transformer_units = [
    projection_dim,
    projection_dim,
]


def CCT(
    input_shape,
    num_heads=2,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
):
    inputs = layers.Input(input_shape)

    # Encode patches.
    cct_tokenizer = CCTTokenizer()
    encoded_patches = cct_tokenizer(inputs)

    # Apply positional embedding.
    if positional_emb:
        sequence_length = encoded_patches.shape[1]
        encoded_patches += PositionEmbedding(sequence_length=sequence_length)(
            encoded_patches
        )

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

    # Classify outputs.
    x = Flatten()(representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=x)
    return model
