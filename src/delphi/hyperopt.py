import os
from pathlib import Path

import click
import darts.metrics
import darts.models
import numpy as np
import optuna
from scipy import stats
import torch
import yaml

from delphi.core.config import UserConfig
from delphi.core.data import TSData
from delphi.train import (
    calculate_loss,
    determine_device,
    determine_model_category,
    forecast,
    prepare_model_params,
    setup_data,
    train_model,
)
from delphi.utils.helpers import create_subdir
from delphi.utils.lightning import PyTorchLightningPruningCallback


def save_aggregated_results(trials, output_path: str | Path, top_n: int = 5):
    # Sort trials by their value and take the top n trials
    best_trials = sorted(trials, key=lambda x: x.value)[:top_n]

    # Create a dictionary to store the aggregated results
    aggregated_results = {}

    # Iterate over the top trials and update the aggregated results dictionary
    for trial in best_trials:
        for k, v in trial.params.items():
            v = int(v) if isinstance(v, bool) else v
            if k not in aggregated_results:
                aggregated_results[k] = {
                    "values": [],
                    "mean": None,
                    "mode": None,
                    "median": None,
                    "std": None,
                }
            aggregated_results[k]["values"].append(v)

    # Calculate the mean, mode, median, and standard deviation for each hyperparameter
    for k, v in aggregated_results.items():
        aggregated_results[k]["mean"] = float(np.mean(v["values"]))
        aggregated_results[k]["mode"] = float(stats.mode(v["values"])[0][0])
        aggregated_results[k]["median"] = float(np.median(v["values"]))
        aggregated_results[k]["std"] = float(np.std(v["values"]))

    # Save the aggregated results to a YAML file
    if isinstance(output_path, str):
        output_path = Path(output_path)
    with open(output_path, "w") as f:
        yaml.dump(aggregated_results, f)

    print(f"Saved aggregated parameters from {top_n} best runs to: {output_path}")


def hyper_opt_params_space_tft(trial):
    model_params = {
        "input_chunk_length": trial.suggest_int("input_chunk_length", 2, 10),
        "output_chunk_length": trial.suggest_int("output_chunk_length", 1, 6),
        "hidden_size": trial.suggest_categorical("hidden_size", [8, 16, 32, 64, 128]),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 4),
        "full_attention": trial.suggest_categorical("full_attention", [True, False]),
        "num_attention_heads": trial.suggest_int("num_attention_heads", 2, 6),
        "hidden_continuous_size": trial.suggest_categorical(
            "hidden_continuous_size", [2, 4, 8, 16, 32, 64]
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
    }
    trainer_params = {
        "batch_size": trial.suggest_int("batch_size", 16, 128),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }
    return model_params, trainer_params


def hyper_opt_params_space_xgb(trial):
    model_params = {
        "lags": trial.suggest_int("lags", 2, 12),
        "lags_past_covariates": trial.suggest_int("lags_past_covariates", 1, 12),
        "lags_future_covariates": trial.suggest_categorical(
            "lags_future_covariates", [[2, 10], [4, 8], [6, 6], [8, 4], [10, 2]]
        ),
        "output_chunk_length": trial.suggest_int("output_chunk_length", 1, 6),
    }
    return model_params, None


def optimize(configs: UserConfig, data: TSData, metric: darts.metrics, n_trials: int = 100):
    hparams = configs.hparams

    # ================================== define objective function================================
    def objective(trial):
        # read from hyper_opt_tft.yaml:

        # select input and output chunk lengths
        mp, tp = hyper_opt_params_space_tft(trial)
        # overwrite values from config with new random values
        hparams.model_params = hparams.model_params | mp
        hparams.trainer_params = hparams.trainer_params | tp
        hparams.model_category = determine_model_category(hparams.model)
        # forcing max epochs to 50
        hparams.trainer_params["n_epochs"] = 50

        # reproducibility
        torch.manual_seed(42)
        hparams.manual_seed = 42

        # throughout training we'll monitor the validation loss for both pruning and early stopping
        pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")

        # Prepare and train the model with new set of parameters
        model_params = prepare_model_params(hparams, data, hparams.device)
        # adding pruner to callbacks
        model_params["pl_trainer_kwargs"]["callbacks"].append(pruner)
        model_class = getattr(darts.models, hparams.model)
        model = model_class(**model_params)
        model = train_model(model, data, hparams)

        # reload best model over course of training
        model = model_class.load_from_checkpoint(hparams.model_name, work_dir=hparams.model_dir)
        # forecast the test set
        y_hat = forecast(model, data, configs)
        # loss calculation
        loss = calculate_loss(data.y_test_ts, y_hat, hparams.metric, n_jobs=hparams.n_jobs)

        return loss if not np.isnan(loss) else 999.0

    # for convenience, print some optimization trials information
    def print_callback(study, trial):
        print(f"Best value: {study.best_value}, Best params: {study.best_params}")

    # optimize hyperparameters by minimizing the MAPE on the validation set
    optuna_logs_dir = create_subdir(hparams.project_dir, "optuna_logs")
    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{str(optuna_logs_dir)}/db.sqlite3",
        study_name=hparams.model_name,
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[print_callback])
    print("Optimization done.")
    return study


# #########################    MAIN SCRIPT   ###############################
@click.command()
@click.option(
    "-c",
    "--config",
    help="Path to config.yaml file.",
    type=str,
    default=os.getcwd() + r"\darts_config.yaml",
)
@click.option(
    "--cpu", help="Train on CPU regardless of GPU availability.", is_flag=True, default=False
)
@click.option(
    "-n",
    "--n_trials",
    help="Number of trial runs passed to optuna's study.optimize()",
    type=int,
    default=200,
)
@click.option(
    "-m",
    "--metric",
    help="Loss function.",
    type=str,
    default="mape",
)
def main(config: str, cpu: bool, n_trials: int, metric: str):
    configs = UserConfig(config)
    configs.create_model_name()
    print(f"Optimizing model {configs.hparams.model_name} to minimize {metric}.")

    # Determine device -- may use cpu even though gpu available
    configs.hparams.device = determine_device() if not cpu else "cpu"

    # configure data, pipelines and model
    configs, data, _ = setup_data(configs)
    study = optimize(configs, data, metric, n_trials)

    # save best config to disk
    hparams = configs.hparams
    model_dir = Path(hparams.model_dir)
    configs.overwrite_hparams_from_dict(study.best_params)
    config_file_path = model_dir.joinpath(f"config_hyperopt_{hparams.model}.yaml")
    configs.save_copy(config_file_path)
    print(f"Saved best config to: {config_file_path}")
    best_params_file = model_dir.joinpath(f"config_hyperopt_{hparams.model}_aggregated.yaml")
    save_aggregated_results(study.trials, best_params_file)


if __name__ == "__main__":
    main()
