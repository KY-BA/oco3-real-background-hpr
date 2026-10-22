# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import logging
import os
import random
import sys
import time

import hydra
import numpy as np
import tensorflow as tf
try:
    import wandb  # type: ignore
    _HAS_WANDB = True
except Exception:
    wandb = None  # type: ignore
    _HAS_WANDB = False
from omegaconf import DictConfig, OmegaConf

from include.callbacks import initiate_wb
from model_training import Model_training_manager


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


@hydra.main(config_path="cfg", config_name="config")
def main_train(cfg: DictConfig):
    logging.basicConfig(
        filename="app.log",
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - [In %(funcName)s: %(filename)s (line %(lineno)d)]",
    )

    print("\n \n \n \n \n Run begins \n \n \n \n \n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    set_global_seed(int(cfg.seed))
    print(f"Global seed set to: {int(cfg.seed)}")
    logging.info("Global seed set to: %d", int(cfg.seed))
    initiate_wb(cfg)
    run_start_time = time.perf_counter()
    model_trainer = Model_training_manager(cfg)
    val_loss = model_trainer.run()
    model_trainer.save()
    training_elapsed_seconds = model_trainer.get_training_elapsed_seconds()
    total_elapsed_seconds = time.perf_counter() - run_start_time
    print(f"Training elapsed time: {format_duration(training_elapsed_seconds)} ({training_elapsed_seconds:.1f} s)")
    print(f"Total run elapsed time: {format_duration(total_elapsed_seconds)} ({total_elapsed_seconds:.1f} s)")
    logging.info(
        "Training elapsed time: %s (%.1f s)",
        format_duration(training_elapsed_seconds),
        training_elapsed_seconds,
    )
    logging.info(
        "Total run elapsed time: %s (%.1f s)",
        format_duration(total_elapsed_seconds),
        total_elapsed_seconds,
    )
    print("\n \n \n \n \n Run ends \n \n \n \n \n")
    if _HAS_WANDB:
        wandb.finish()  # type: ignore
    return val_loss


if __name__ == "__main__":
    main_train()
