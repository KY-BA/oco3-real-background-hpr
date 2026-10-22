# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
import tensorflow as tf
from hydra import compose, initialize
from hydra.utils import call, instantiate
from icecream import ic
from omegaconf import DictConfig, OmegaConf

import include.generators as generators
from Data import Data_train


@dataclass
class Trainer:
    """Train Convolutional Neural Networks models."""

    generator: tf.keras.utils.Sequence
    validation_data: Any = None
    callbacks: list = field(default_factory=lambda: [])
    batch_size: int = 32
    max_epochs: int = 10
    training_elapsed_seconds: float = 0.0

    def train_model(self, model: tf.keras.Model, data: Data_train) -> tf.keras.Model:
        """Train model and evaluate validation."""
        start_time = time.perf_counter()
        self.history = model.fit(
            self.generator,
            epochs=self.max_epochs,
            validation_data=self.validation_data,
            verbose="auto",
            steps_per_epoch=len(self.generator),
            callbacks=self.callbacks,
            shuffle=True,
        )
        self.training_elapsed_seconds = time.perf_counter() - start_time

        return model

    def get_history(self):
        """Return history."""
        return self.history

    def get_val_loss(self):
        """Return best val loss."""
        return np.min(self.history.history["val_loss"])

    def get_training_elapsed_seconds(self):
        """Return total model.fit wall-clock time in seconds."""
        return float(self.training_elapsed_seconds)
