# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2024
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from tensorflow import keras

# Make wandb optional
try:  # type: ignore
    from wandb.keras import WandbCallback  # type: ignore
    import wandb  # type: ignore
    _HAS_WANDB = True
except Exception:  # pragma: no cover - optional dependency
    WandbCallback = None  # type: ignore
    wandb = None  # type: ignore
    _HAS_WANDB = False


def _cfg_enabled(cfg_or_bool) -> bool:
    if isinstance(cfg_or_bool, bool):
        return cfg_or_bool
    if cfg_or_bool is None:
        return False
    for key in ["__target__", "_target_", "enabled"]:
        if hasattr(cfg_or_bool, key):
            value = getattr(cfg_or_bool, key)
            if value is not None:
                return bool(value)
    return True


def initiate_wb(cfg: DictConfig) -> None:
    """Initiate Weights & Biases, if available and enabled."""
    if not getattr(cfg.callbacks, "wandb", False) or not _HAS_WANDB:
        return

    config_wb = {
        "train": cfg.data.path.train.name,
        "valid": cfg.data.path.valid.name,
        "model": cfg.model.name,
        "chan_0": cfg.data.input.chan_0,
        "chan_1": cfg.data.input.chan_1,
        "chan_2": cfg.data.input.chan_2,
        "chan_3": cfg.data.input.chan_3,
        "chan_4": cfg.data.input.chan_4,
        "clouds_threshold": cfg.data.input.clouds_threshold,
        "loss": cfg.training.loss_func,
        "scaling_method": cfg.augmentations.scaling_method,
        "timedate": cfg.data.input.timedate,
        "init_lr": cfg.training.init_lr,
        "dropout_rate": cfg.model.dropout_rate,
    }

    if cfg.sweep:
        wandb.init(  # type: ignore
            project=cfg.exp_name,
            config=config_wb,
            name=os.path.basename(os.getcwd()),
            settings=wandb.Settings(start_method="thread"),  # type: ignore
        )
    else:
        wandb.init(  # type: ignore
            project=cfg.model.type,
            config=config_wb,
            settings=wandb.Settings(start_method="thread"),  # type: ignore
        )


def get_modelcheckpoint(get: bool, cbs: list, filepath="w_best.h5") -> list:
    """Add modelcheckpoint to callbacks list if get."""
    if _cfg_enabled(get):
        modelcheckpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_weights_only=False,
            monitor=getattr(get, "monitor", "val_loss"),
            mode=getattr(get, "mode", "auto"),
            save_best_only=True,
            verbose=1,
        )
        cbs.append(modelcheckpoint_cb)
    else:
        pass
    return cbs


def get_lrscheduler(get: bool, cbs: list) -> list:
    """Add reducelronplateau to callbacks list if get."""
    if _cfg_enabled(get):
        reducelronplateau_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=getattr(get, "monitor", "val_loss"),
            factor=getattr(get, "factor", 0.5),
            patience=getattr(get, "patience", 20),
            verbose=getattr(get, "verbose", 0),
            min_delta=getattr(get, "min_delta", 5e-3),
            cooldown=getattr(get, "cooldown", 0),
            min_lr=getattr(get, "min_lr", 5e-5),
        )
        cbs.append(reducelronplateau_cb)
    else:
        pass
    return cbs


def get_earlystopping(get: bool, cbs: list) -> list:
    """Add earlystopping to callbacks list if get."""
    if _cfg_enabled(get):
        earlystopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor=getattr(get, "monitor", "val_loss"),
            min_delta=getattr(get, "min_delta", 1e-3),
            patience=getattr(get, "patience", 60),
            verbose=getattr(get, "verbose", 0),
            mode=getattr(get, "mode", "auto"),
            baseline=getattr(get, "baseline", None),
            restore_best_weights=getattr(get, "restore_best_weights", True),
        )
        cbs.append(earlystopping_cb)
    else:
        pass
    return cbs


def get_wandb(get: bool, cbs: list) -> list:
    """Add wandb to callbacks list if get and available."""
    if _cfg_enabled(get) and _HAS_WANDB and WandbCallback is not None:
        cbs.append(WandbCallback())  # type: ignore
    return cbs


def get_csvlogger(get: bool, cbs: list, filename: str = "training_log.csv") -> list:
    """Add CSVLogger to persist epoch-level training metrics."""
    if _cfg_enabled(get):
        cbs.append(
            tf.keras.callbacks.CSVLogger(
                filename=getattr(get, "filename", filename),
                separator=getattr(get, "separator", ","),
                append=getattr(get, "append", False),
            )
        )
    return cbs


class ExtraValidation(tf.keras.callbacks.Callback):
    def __init__(self, extra_val_data, name):
        super(ExtraValidation, self).__init__()

        self.extra_val_data = extra_val_data
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        if isinstance(self.extra_val_data, tuple) and len(self.extra_val_data) == 2:
            (extra_val_data, extra_val_targets) = self.extra_val_data
            extra_val_loss = self.model.evaluate(
                extra_val_data, extra_val_targets, verbose=0
            )
        else:
            extra_val_loss = self.model.evaluate(self.extra_val_data, verbose=0)
        print(f"{self.name}_loss:", extra_val_loss)

        # Log to wandb only if available and explicitly initialized.
        if _HAS_WANDB and wandb is not None and getattr(wandb, "run", None) is not None:
            if isinstance(extra_val_loss, list):
                wandb.log({f"{self.name}_loss": extra_val_loss[0]})  # type: ignore
                for idx, metric in enumerate(extra_val_loss[1:]):
                    wandb.log({f"{self.name}_metric_{idx}": metric})  # type: ignore
            else:
                wandb.log({f"{self.name}_loss": extra_val_loss})  # type: ignore
