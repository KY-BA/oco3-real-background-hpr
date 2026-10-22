# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2024
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf
from hydra import compose, initialize
from hydra.utils import call, instantiate
from icecream import ic
from omegaconf import DictConfig, OmegaConf

try:
    import models.reg as rm
except ImportError:
    pass

# Make wandb optional for metrics logger
try:  # type: ignore
    from wandb.keras import WandbMetricsLogger  # type: ignore
    _HAS_WANDB = True
except Exception:  # pragma: no cover
    from tensorflow import keras as _keras

    class WandbMetricsLogger(_keras.callbacks.Callback):  # type: ignore
        pass

    _HAS_WANDB = False

import include.callbacks as callbacks
import include.generators as generators
import include.loss as loss
import include.optimisers as optimisers
import models.seg as sm
from Data import Data_train
from include.trainer import Trainer
from models.preprocessing import CloudsLayer
from saver import Saver


class Model_training_manager:
    """Manager for segmentation, inversion with CNN models."""

    def __init__(self, cfg: DictConfig) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info("Starting data preparation...")
        self.prepare_data(cfg)
        logging.info("Data preparation completed.")

        logging.info("Starting model building...")
        self.build_model(cfg)
        self.compile_model(cfg)
        logging.info("Model building completed.")

        logging.info("Starting training preparation...")
        self.prepare_training(cfg)
        logging.info("Training preparation completed.")

        logging.info("Initializing the saver...")
        self.saver = Saver()
        logging.info("Saver initialized.")

    def prepare_data(self, cfg: DictConfig) -> None:
        """Prepare Data inputs to the neural network and outputs (=labels, targets)."""

        print(cfg.data.init.path_train_ds)
        self.data = instantiate(cfg.data.init)

        if cfg.model.type in ["inversion", "uq"]:
            self.data.prepare_input(
                chan_0=getattr(cfg.data.input, "chan_0", "None"),
                chan_1=getattr(cfg.data.input, "chan_1", "None"),
                chan_2=getattr(cfg.data.input, "chan_2", "None"),
                chan_3=getattr(cfg.data.input, "chan_3", "None"),
                chan_4=getattr(cfg.data.input, "chan_4", "None"),
                clouds_threshold=getattr(cfg.data.input, "clouds_threshold", 0.5),
                dir_clouds_array=getattr(cfg.data.input, "dir_clouds_array", "None"),
                dir_seg_models=getattr(cfg.data.input, "dir_seg_models", "None"),
                timedate=getattr(cfg.data.input, "timedate", False),
                field_name=getattr(cfg.data.input, "field_name", "xco2_mixed"),
            )
            self.data.prepare_output_inversion(cfg.data.output.N_emissions)
            if cfg.model.name in [
                "bg_plume_aux_cnn_tiny",
                "supervised_plume_attention_cnn_tiny",
                "attention_feature_aux_v2_cnn_tiny",
            ]:
                aux_field = getattr(cfg.data.output, "aux_field", "plume_aux")
                self.data.y.load_auxiliary_field(aux_field)
        else:
            logging.info("No model type selected.")
            sys.exit(1)

    def build_model(self, cfg: DictConfig) -> None:
        """Build the inversion or segmentation model."""

        if "load_weights" in cfg:
            self.model = tf.keras.models.load_model(
                os.path.join(cfg.dir_res, cfg.load_weights, "w_last.h5"),
                compile=True,
            )

        else:
            if cfg.model.type in ["inversion", "uq"]:
                try:
                    reg_builder = instantiate(
                        cfg.model.init,
                        input_shape=self.data.x.fields_input_shape,
                        classes=self.data.y.classes,
                        norm_layer=self.data.x.norm_layer,
                        noise_layer=self.data.x.noise_layer,
                        cloud_layer=self.data.x.cloud_layer,
                        timedate_vector_size=self.data.x.timedate_vector_size,
                        set_size=getattr(cfg.model, "set_size", 4),
                    )

                    self.model = reg_builder.get_model()
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    sys.exit(1)
            else:
                logging.info("No model type selected.")
                sys.exit(1)

    def compile_model(self, cfg: DictConfig) -> None:
        try:
            if cfg.model.name in [
                "bg_plume_aux_cnn_tiny",
                "supervised_plume_attention_cnn_tiny",
                "attention_feature_aux_v2_cnn_tiny",
            ]:
                self.model.compile(
                    optimizer=optimisers.define_optimiser(
                        cfg.training.optimiser, cfg.training.init_lr
                    ),
                    loss={
                        "emission": loss.define_loss(cfg.training.loss_func),
                        "plume_aux": tf.keras.losses.MeanSquaredError(),
                    },
                    loss_weights={
                        "emission": 1.0,
                        "plume_aux": getattr(cfg.training, "aux_loss_weight", 0.05),
                    },
                    metrics={
                        "emission": loss.define_metrics(cfg.model.type),
                        "plume_aux": [tf.keras.metrics.MeanSquaredError(name="mse")],
                    },
                )
                return
            if cfg.model.name in ["scale_aware_cnn_tiny", "scale_decomposed_cnn_tiny"]:
                self.model.compile(
                    optimizer=optimisers.define_optimiser(
                        cfg.training.optimiser, cfg.training.init_lr
                    ),
                    loss={
                        "emission": loss.define_loss(cfg.training.loss_func),
                        "emission_bin": tf.keras.losses.CategoricalCrossentropy(),
                    },
                    loss_weights={
                        "emission": 1.0,
                        "emission_bin": getattr(cfg.training, "bin_loss_weight", 0.1),
                    },
                    metrics={
                        "emission": loss.define_metrics(cfg.model.type),
                        "emission_bin": [
                            tf.keras.metrics.CategoricalAccuracy(
                                name="accuracy"
                            )
                        ],
                    },
                )
                return
            metrics = (
                []
                if cfg.training.loss_func == "ScaleAwarePerturbationConsistentMAE"
                else loss.define_metrics(cfg.model.type)
            )
            self.model.compile(
                optimizer=optimisers.define_optimiser(
                    cfg.training.optimiser, cfg.training.init_lr
                ),
                loss=loss.define_loss(cfg.training.loss_func),
                metrics=metrics,
            )
        except Exception as e:
            logging.error(f"Compilation error: {e}")
            sys.exit(1)

    def prepare_training(self, cfg: DictConfig) -> None:
        """Prepare Trainer object."""
        generator = self.prepare_generator(cfg)
        validation_data = self.prepare_validation_data(cfg)
        callbacks = self.prepare_callbacks(cfg)

        try:
            self.trainer = Trainer(
                generator=generator,
                validation_data=validation_data,
                callbacks=callbacks,
                batch_size=cfg.training.batch_size,
                max_epochs=cfg.training.max_epochs,
            )

        except Exception as e:
                logging.error(f"An error occurred at Trainer preparation: {e}")
                sys.exit(1)

    def prepare_generator(self, cfg: DictConfig) -> tf.keras.utils.Sequence:
        """Prepare generator to feed to Trainer."""
        ic(cfg.model.type)
        if getattr(self.data.x, "precomputed", False):
            try:
                if cfg.model.name == "observation_set_encoder_tiny":
                    generator = generators.PrecomputedSetDataGen(
                        self.data.x.train,
                        self.data.y.train,
                        batch_size=cfg.training.batch_size,
                        shuffle=getattr(cfg.augmentations, "shuffle", True),
                        set_size=getattr(cfg.model, "set_size", 4),
                        seed=getattr(cfg, "seed", 42),
                    )
                elif cfg.model.name in [
                    "bg_plume_aux_cnn_tiny",
                    "supervised_plume_attention_cnn_tiny",
                    "attention_feature_aux_v2_cnn_tiny",
                ]:
                    generator = generators.PrecomputedMultiTaskDataGen(
                        self.data.x.train,
                        self.data.y.train,
                        self.data.y.plume_train,
                        batch_size=cfg.training.batch_size,
                        shuffle=getattr(cfg.augmentations, "shuffle", True),
                    )
                elif cfg.model.name in ["scale_aware_cnn_tiny", "scale_decomposed_cnn_tiny"]:
                    generator = generators.PrecomputedScaleAwareDataGen(
                        self.data.x.train,
                        self.data.y.train,
                        batch_size=cfg.training.batch_size,
                        shuffle=getattr(cfg.augmentations, "shuffle", True),
                        bin_balanced=getattr(cfg.augmentations, "bin_balanced", True),
                        low_threshold=getattr(cfg.augmentations, "low_threshold", 8.0),
                        mid_threshold=getattr(cfg.augmentations, "mid_threshold", 18.0),
                        high_threshold=getattr(
                            cfg.augmentations, "high_threshold", 30.0
                        ),
                    )
                elif getattr(cfg.augmentations, "counterfactual", False):
                    generator = generators.PrecomputedCounterfactualDataGen(
                        self.data.x.train,
                        self.data.y.train,
                        self.data.path_train_h5,
                        batch_size=cfg.training.batch_size,
                        shuffle=getattr(cfg.augmentations, "shuffle", True),
                        include_singletons=getattr(
                            cfg.augmentations, "include_singletons", True
                        ),
                    )
                elif getattr(cfg.augmentations, "consistency_noise", False):
                    generator = generators.PrecomputedConsistencyNoiseDataGen(
                        self.data.x.train,
                        self.data.y.train,
                        batch_size=cfg.training.batch_size,
                        shuffle=getattr(cfg.augmentations, "shuffle", True),
                        norm_noise_std_min=getattr(
                            cfg.augmentations, "norm_noise_std_min", 0.0
                        ),
                        norm_noise_std_max=getattr(
                            cfg.augmentations, "norm_noise_std_max", 0.3
                        ),
                        low_threshold=getattr(cfg.augmentations, "low_threshold", 8.0),
                    )
                elif getattr(cfg.augmentations, "weighted", False):
                    generator = generators.PrecomputedWeightedDataGen(
                        self.data.x.train,
                        self.data.y.train,
                        batch_size=cfg.training.batch_size,
                        shuffle=getattr(cfg.augmentations, "shuffle", True),
                        high_threshold=getattr(
                            cfg.augmentations, "high_threshold", 18.0
                        ),
                        very_high_threshold=getattr(
                            cfg.augmentations, "very_high_threshold", 30.0
                        ),
                        high_weight=getattr(cfg.augmentations, "high_weight", 1.5),
                        very_high_weight=getattr(
                            cfg.augmentations, "very_high_weight", 2.0
                        ),
                    )
                elif getattr(cfg.augmentations, "bin_balanced", False):
                    generator = generators.PrecomputedBinBalancedDataGen(
                        self.data.x.train,
                        self.data.y.train,
                        batch_size=cfg.training.batch_size,
                        shuffle=getattr(cfg.augmentations, "shuffle", True),
                        low_threshold=getattr(cfg.augmentations, "low_threshold", 8.0),
                        mid_threshold=getattr(cfg.augmentations, "mid_threshold", 18.0),
                        high_threshold=getattr(
                            cfg.augmentations, "high_threshold", 30.0
                        ),
                        norm_noise_std_min=getattr(
                            cfg.augmentations, "norm_noise_std_min", 0.0
                        ),
                        norm_noise_std_max=getattr(
                            cfg.augmentations, "norm_noise_std_max", 0.0
                        ),
                    )
                else:
                    generator = generators.PrecomputedDataGen(
                        self.data.x.train,
                        self.data.y.train,
                        batch_size=cfg.training.batch_size,
                        shuffle=getattr(cfg.augmentations, "shuffle", True),
                        norm_noise_std_min=getattr(
                            cfg.augmentations, "norm_noise_std_min", 0.0
                        ),
                        norm_noise_std_max=getattr(
                            cfg.augmentations, "norm_noise_std_max", 0.0
                        ),
                    )
            except Exception as e:
                logging.error(f"An error occurred at precomputed generator preparation: {e}")
                sys.exit(1)
            return generator

        if cfg.model.type == "inversion":
            try:
                generator = instantiate(
                    cfg.augmentations,
                    self.data.x.train,
                    self.data.x.xco2_plumes_train,
                    self.data.x.xco2_back_train,
                    self.data.x.xco2_alt_anthro_train,
                    getattr(self.data.x, "no2_plumes_train", None),
                    getattr(self.data.x, "no2_back_train", None),
                    self.data.y.train,
                    self.data.x.list_chans,
                    self.data.x.clouds_training,
                    self.data.x.noise_layer,
                    self.data.x.norm_layer,
                    self.data.x.cloud_layer,
                )
            except AttributeError as e:
                logging.error(f"Missing attribute at generator preparation: {e}")
                sys.exit(1)
            except Exception as e:
                logging.error(f"An error occurred at generator preparation: {e}")
                sys.exit(1)
        else:
            logging.info("No model type selected...")
            sys.exit(1)
        return generator

    def prepare_validation_data(self, cfg: DictConfig):
        """Prepare validation input for model.fit."""
        if getattr(self.data.x, "precomputed", False):
            try:
                if cfg.model.name == "observation_set_encoder_tiny":
                    validation_data = generators.PrecomputedSetDataGen(
                        self.data.x.valid,
                        self.data.y.valid,
                        batch_size=cfg.training.batch_size,
                        shuffle=False,
                        set_size=getattr(cfg.model, "set_size", 4),
                        seed=getattr(cfg, "seed", 42),
                    )
                elif cfg.model.name in [
                    "bg_plume_aux_cnn_tiny",
                    "supervised_plume_attention_cnn_tiny",
                    "attention_feature_aux_v2_cnn_tiny",
                ]:
                    validation_data = generators.PrecomputedMultiTaskDataGen(
                        self.data.x.valid,
                        self.data.y.valid,
                        self.data.y.plume_valid,
                        batch_size=cfg.training.batch_size,
                        shuffle=False,
                    )
                elif cfg.model.name in ["scale_aware_cnn_tiny", "scale_decomposed_cnn_tiny"]:
                    validation_data = generators.PrecomputedScaleAwareDataGen(
                        self.data.x.valid,
                        self.data.y.valid,
                        batch_size=cfg.training.batch_size,
                        shuffle=False,
                        bin_balanced=False,
                        low_threshold=getattr(cfg.augmentations, "low_threshold", 8.0),
                        mid_threshold=getattr(cfg.augmentations, "mid_threshold", 18.0),
                        high_threshold=getattr(
                            cfg.augmentations, "high_threshold", 30.0
                        ),
                    )
                elif getattr(cfg.augmentations, "counterfactual", False):
                    validation_data = generators.PrecomputedCounterfactualDataGen(
                        self.data.x.valid,
                        self.data.y.valid,
                        self.data.path_valid_h5,
                        batch_size=cfg.training.batch_size,
                        shuffle=False,
                        include_singletons=True,
                    )
                else:
                    validation_data = generators.PrecomputedDataGen(
                        self.data.x.valid,
                        self.data.y.valid,
                        batch_size=cfg.training.batch_size,
                        shuffle=False,
                    )
            except Exception as e:
                logging.error(
                    f"An error occurred at precomputed validation generator preparation: {e}"
                )
                sys.exit(1)
            return validation_data

        return (self.data.x.valid, self.data.y.valid)

    def prepare_callbacks(self, cfg: DictConfig) -> list:
        """Prepare list of callbacks to feed to Trainer."""
        cbs = callbacks.get_modelcheckpoint(cfg.callbacks.model_checkpoint, [])
        cbs = callbacks.get_lrscheduler(cfg.callbacks.learning_rate_monitor, cbs)
        if hasattr(cfg.callbacks, "early_stopping"):
            cbs = callbacks.get_earlystopping(cfg.callbacks.early_stopping, cbs)
        cbs = callbacks.get_csvlogger(
            getattr(cfg.callbacks, "csv_logger", True),
            cbs,
        )
        cbs = callbacks.get_wandb(cfg.callbacks.wandb, cbs)

        if getattr(self.data.x, "precomputed", False):
            if cfg.model.name == "observation_set_encoder_tiny":
                valid_2_data = generators.PrecomputedSetDataGen(
                    self.data.x.valid_2,
                    self.data.y.valid_2,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    set_size=getattr(cfg.model, "set_size", 4),
                    seed=getattr(cfg, "seed", 42),
                )
                extra_valid_data = generators.PrecomputedSetDataGen(
                    self.data.x.extra_valid,
                    self.data.y.extra_valid,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    set_size=getattr(cfg.model, "set_size", 4),
                    seed=getattr(cfg, "seed", 42),
                )
            elif cfg.model.name in [
                "bg_plume_aux_cnn_tiny",
                "supervised_plume_attention_cnn_tiny",
                "attention_feature_aux_v2_cnn_tiny",
            ]:
                valid_2_data = generators.PrecomputedMultiTaskDataGen(
                    self.data.x.valid_2,
                    self.data.y.valid_2,
                    self.data.y.plume_valid_2,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                )
                extra_valid_data = generators.PrecomputedMultiTaskDataGen(
                    self.data.x.extra_valid,
                    self.data.y.extra_valid,
                    self.data.y.plume_extra_valid,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                )
            elif cfg.model.name in ["scale_aware_cnn_tiny", "scale_decomposed_cnn_tiny"]:
                valid_2_data = generators.PrecomputedScaleAwareDataGen(
                    self.data.x.valid_2,
                    self.data.y.valid_2,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    bin_balanced=False,
                    low_threshold=getattr(cfg.augmentations, "low_threshold", 8.0),
                    mid_threshold=getattr(cfg.augmentations, "mid_threshold", 18.0),
                    high_threshold=getattr(cfg.augmentations, "high_threshold", 30.0),
                )
                extra_valid_data = generators.PrecomputedScaleAwareDataGen(
                    self.data.x.extra_valid,
                    self.data.y.extra_valid,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    bin_balanced=False,
                    low_threshold=getattr(cfg.augmentations, "low_threshold", 8.0),
                    mid_threshold=getattr(cfg.augmentations, "mid_threshold", 18.0),
                    high_threshold=getattr(cfg.augmentations, "high_threshold", 30.0),
                )
            elif getattr(cfg.augmentations, "counterfactual", False):
                valid_2_data = generators.PrecomputedCounterfactualDataGen(
                    self.data.x.valid_2,
                    self.data.y.valid_2,
                    self.data.path_valid_2_h5,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    include_singletons=True,
                )
                extra_valid_data = generators.PrecomputedCounterfactualDataGen(
                    self.data.x.extra_valid,
                    self.data.y.extra_valid,
                    self.data.path_extra_valid_h5,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    include_singletons=True,
                )
            else:
                valid_2_data = generators.PrecomputedDataGen(
                    self.data.x.valid_2,
                    self.data.y.valid_2,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                )
                extra_valid_data = generators.PrecomputedDataGen(
                    self.data.x.extra_valid,
                    self.data.y.extra_valid,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                )
        else:
            valid_2_data = (self.data.x.valid_2, self.data.y.valid_2)
            extra_valid_data = (self.data.x.extra_valid, self.data.y.extra_valid)

        history_2 = callbacks.ExtraValidation(valid_2_data, "valid_2")
        history = callbacks.ExtraValidation(extra_valid_data, "extra_valid")

        # Only attach metrics logger if wandb enabled and available
        if getattr(cfg.callbacks, "wandb", False) and _HAS_WANDB:
            cbs.append(WandbMetricsLogger())
        cbs.append(history_2)
        cbs.append(history)
        return cbs

    def run(self) -> None:
        """Train the model with the training data."""
        try:
            self.model = self.trainer.train_model(self.model, self.data)
        except Exception as e:
            logging.error(f"An error occurred at RUN time: {e}")
            sys.exit(1)
        return self.trainer.get_val_loss()

    def get_training_elapsed_seconds(self) -> float:
        """Return total model.fit wall-clock time in seconds."""
        return self.trainer.get_training_elapsed_seconds()

    def save(self) -> None:
        """Save results of the run."""
        print("Saving at:", os.getcwd())
        self.saver.save_norm_layer(
            self.data.x.norm_layer, self.data.x.fields_input_shape
        )
        self.saver.save_cloud_layer(
            self.data.x.cloud_layer, self.data.x.fields_input_shape
        )
        self.saver.save_model_and_weights(self.model)
