"""
Hyperparameter search with Optuna.

Is Optuna actually worth it here? Honest answer: for a single DistilBERT run on a
free-tier GPU, the marginal gain from tuning is usually small (a point or two of
macro-F1) compared to a well-chosen default config -- but demonstrating a *principled*
search (defined search space, pruning, objective tied to macro-F1 rather than
accuracy) is exactly the kind of engineering practice that separates a resume
project from a tutorial. So: yes, include it, but keep the budget small and use
pruning so bad trials die early instead of burning your GPU quota.

- Search space: learning rate, weight decay, warmup ratio, batch size, epochs
  (see configs/config.yaml -> optuna.search_space).
- Objective: validation macro-F1 (not accuracy -- the dataset has class imbalance,
  and macro-F1 penalizes ignoring the minority class).
- Trials: 15 by default. That's a deliberately small, free-GPU-friendly budget;
  raise `optuna.n_trials` in the config if you have more compute.
- Pruning: MedianPruner stops clearly-losing trials after the first eval epoch.
"""
import argparse
import json
from pathlib import Path

import optuna
import yaml
from optuna.pruners import MedianPruner

from src.training.train_transformer import train


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_objective(cfg: dict):
    space = cfg["optuna"]["search_space"]

    def objective(trial: optuna.Trial) -> float:
        overrides = {
            "learning_rate": trial.suggest_float(
                "learning_rate", *space["learning_rate"], log=True
            ),
            "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"]),
            "warmup_ratio": trial.suggest_float("warmup_ratio", *space["warmup_ratio"]),
            "batch_size": trial.suggest_categorical(
                "batch_size", space["per_device_train_batch_size"]
            ),
            "num_train_epochs": trial.suggest_int(
                "num_train_epochs", *space["num_train_epochs"]
            ),
            "output_dir": f"models/optuna_trials/trial_{trial.number}",
        }
        trainer = train(cfg, overrides=overrides)
        metrics = trainer.evaluate()
        return metrics["eval_f1_macro"]

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ocfg = cfg["optuna"]

    study = optuna.create_study(
        study_name=ocfg["study_name"],
        direction=ocfg["direction"],
        pruner=MedianPruner(n_warmup_steps=1),
    )
    study.optimize(make_objective(cfg), n_trials=ocfg["n_trials"])

    print("Best trial:")
    print(f"  value (eval_f1_macro): {study.best_trial.value}")
    print(f"  params: {study.best_trial.params}")

    out_path = Path("models/optuna_best_params.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"value": study.best_trial.value, "params": study.best_trial.params}, f, indent=2
        )
    print(f"Saved best params to {out_path}")


if __name__ == "__main__":
    main()
