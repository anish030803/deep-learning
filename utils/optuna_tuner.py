"""
Optuna-based automatic hyperparameter tuning.
Triggered when validation AUC falls below configs.config.RETUNE_THRESHOLD.

Searches:
  - learning rate (backbone + head)
  - weight decay (L2)
  - L1 lambda
  - dropout rate
  - batch size
  - warmup epochs
  - label smoothing
  - optimizer choice
"""

import os
import sys
import torch
import numpy as np
import optuna
from optuna.samplers import TPESampler
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, RETUNE_THRESHOLD, SEED,
    NUM_CLASSES, MAX_EPOCHS, CHECKPOINT_DIR,
)


optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_hyperparams(trial: optuna.Trial) -> dict:
    """Define the Optuna search space."""
    return {
        "lr":              trial.suggest_float("lr",           1e-5, 1e-3, log=True),
        "lr_head":         trial.suggest_float("lr_head",      1e-4, 1e-2, log=True),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "l1_lambda":       trial.suggest_float("l1_lambda",    1e-7, 1e-3, log=True),
        "dropout_rate":    trial.suggest_float("dropout_rate", 0.1,  0.5),
        "batch_size":      trial.suggest_categorical("batch_size", [16, 32, 64]),
        "warmup_epochs":   trial.suggest_int("warmup_epochs",  2, 10),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "optimizer":       trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"]),
    }


def run_optuna(
    objective_fn: Callable[[dict], float],
    n_trials:     int   = OPTUNA_N_TRIALS,
    timeout:      int   = OPTUNA_TIMEOUT,
    study_name:   str   = "efficientnet_vindr",
    storage_path: str | None = None,
) -> dict:
    """
    Run Optuna hyperparameter search.

    Args:
        objective_fn: callable(hp_dict) → float (validation AUC, higher is better)
        n_trials    : number of Optuna trials
        timeout     : wall-clock timeout in seconds
        study_name  : Optuna study identifier
        storage_path: path to sqlite DB for persistence (optional)

    Returns:
        best_params dict
    """
    storage = f"sqlite:///{storage_path}" if storage_path else None

    study = optuna.create_study(
        direction   = "maximize",
        sampler     = TPESampler(seed=SEED),
        study_name  = study_name,
        storage     = storage,
        load_if_exists = True,
    )

    def _wrapped_objective(trial: optuna.Trial) -> float:
        hp = suggest_hyperparams(trial)
        try:
            val_auc = objective_fn(hp)
        except Exception as e:
            print(f"[Optuna] Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()
        return val_auc

    print(f"\n{'='*60}")
    print(f"  [Optuna] Starting hyperparameter search")
    print(f"  Trials: {n_trials}  |  Timeout: {timeout}s")
    print(f"  Trigger: val_auc < {RETUNE_THRESHOLD}")
    print(f"{'='*60}\n")

    study.optimize(
        _wrapped_objective,
        n_trials      = n_trials,
        timeout       = timeout,
        show_progress_bar = True,
        gc_after_trial    = True,
    )

    best = study.best_trial
    print(f"\n[Optuna] Best trial #{best.number}")
    print(f"  Best val AUC : {best.value:.4f}")
    print(f"  Best params  :")
    for k, v in best.params.items():
        print(f"    {k:<20} = {v}")

    return best.params


def should_retune(val_auc: float) -> bool:
    """Return True if val_auc is below the retuning threshold."""
    return val_auc < RETUNE_THRESHOLD
