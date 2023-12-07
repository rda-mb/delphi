""" Main training functionalities of delphi. Contains all functions to setup runs, setus
    data, setup model, train, predict, calculate loss and plot the timeseries. """
import atexit
from dataclasses import asdict
import os
from pathlib import Path
import sys
import warnings

import click
import darts
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
import mlflow
import numpy as np

# import darts.models
from pytorch_lightning.callbacks import EarlyStopping
import torch

from delphi.core.config import ConfigSchema, UserConfig
from delphi.core.data import TSData, TSLoader, TSPipeline
from delphi.core.visual import plot_timeseries
from delphi.utils.helpers import Logger, create_subdir
from delphi.utils.lightning import format_pl_trainer_params
from delphi.utils.torch import determine_device


warnings.filterwarnings("ignore", ".*does not have many workers.*")

# ##### PRIORITY
# TODO saved config keep adding comments, shown when using a config file from hyperopt.
# TODO make better retraining code, use load_weights?
# TODO prepare_predict_params()

# ##### NEW FEATURES
# ask blazej if random state passed to Model?
# TODO add pca_corr_cutoff step as sklearn custom function callable in pipeline
# TODO add sklearn.preprocessing.FunctionTransformer for custom preprocessing fn e.g boxcox
# TODO test Darts `LinearRegressionModel`, `RandomForest`
# TODO change inverse_transform to use .mean() and quantile_timeseries()

# #### NOT URGENT
# TODO update how hyperopt params are parsed from yaml file
# TODO generate code docs using sphinx
# TODO setup Makefile
# TODO try to find fix on hydra branch
# TODO branch and try to make a class ForecastingPipeline or Workflow


def calculate_loss(y_true, y_hat, metric, n_jobs):
    loss_fn = getattr(darts.metrics, metric)
    losses = loss_fn(y_true, y_hat, n_jobs=n_jobs, verbose=True)
    return np.mean(losses)


def determine_model_category(model_class: str) -> str:
    category_dict = {
        "TFTModel": "deep learning",
        "TransformerModel": "deep learning",
        "TCNModel": "deep learning",
        "RNNModel": "deep learning",
        "BlockRNNModel": "deep learning",
        "XGBModel": "boosting",
        "LightGBMModel": "boosting",
        "CatBoostModel": "boosting",
        "LinearRegressionModel": "machine learning",
        "RandomForest": "machine learning",
        "AutoARIMA": "statistical",
    }

    return category_dict.get(model_class, "unknown")


def prepare_model_params(hparams: ConfigSchema, data: TSData, device: str) -> dict:
    model_params = dict(hparams.model_params)
    model_params["work_dir"] = hparams.darts_logs_dir

    # for boosting models, ensure future_lags set to None if no future covs
    if hparams.model_category == "boosting":
        if data.X_future is None:
            model_params["lags_future_covariates"] = None

    # likelihood for deep learning model - has to be instantiated
    if model_params["likelihood"] == "QuantileRegression":
        likelihood_fn = darts.utils.likelihood_models.QuantileRegression
        model_params["likelihood"] = likelihood_fn(model_params["quantiles"])
        # remove 'quantiles' from model_params dict
        model_params.pop("quantiles")

    # preprare trainer for Deep Learning models
    if hparams.model_category == "deep learning":
        trainer_params = format_pl_trainer_params(
            hparams.trainer_params, hparams.model_name, device=device
        )
        # append trainer params to model parameters
        model_params = model_params | trainer_params

    return model_params


def prepare_predict_params():
    pass


def setup_data(configs: UserConfig) -> tuple[UserConfig, TSData, TSPipeline]:
    hparams = configs.hparams
    project_dir = Path(hparams.project_dir)
    hparams.darts_logs_dir = str(Path.home().joinpath("darts_logs"))

    # ==================== create directory to store trained model and metadata ==================
    models_dir = create_subdir(project_dir, "trained_models")
    parent_dir = models_dir
    # create nested folder trained_models/variable/category/sub-category/etc...
    if not isinstance(hparams.target, list):
        hparams.target = [hparams.target]
    for v in hparams.target:
        parent_dir = create_subdir(parent_dir, v)

    model_dir = create_subdir(parent_dir, hparams.model_name)
    hparams.model_dir = str(model_dir)

    # ==================================== set up logging ========================================
    stage = hparams.stage
    print("Stage:", stage)
    logger = Logger(model_dir.joinpath(f"log_{stage}.txt"))
    sys.stdout = logger
    sys.stderr = sys.stdout
    atexit.register(logger.close)

    # =========================================== load data ======================================
    loader = TSLoader(hparams.data_sources)
    loader.crop_date_range(hparams.data_date_start, hparams.data_date_stop)
    target_column = loader.validate_target_variable(hparams.target)
    raw_data = loader.trim_data(
        hparams.trim_initial_zeros, hparams.trim_low_corr_variables_upfront
    )

    # ======================================== split the data ====================================
    data = TSData(raw_data, target=target_column, stage=stage)
    data.split_target_from_covariates()
    data.split_past_future_covariates(hparams.force_use_only_past_covs)
    data.split_train_test(hparams.split, hparams.test_set_at_beginning)

    # ====================================== preprocess data =====================================
    # fit and transform target data
    target_pipe = TSPipeline(hparams.target_preprocess, dtype="target")
    target_pipe.fit_or_load(data.y_train, model_dir, stage=stage)
    data.y_train = target_pipe.transform(data.y_train)
    data.y_test = target_pipe.transform(data.y_test)

    # fit and transform past cov data
    if data.past_cov_cols:
        past_cov_pipe = TSPipeline(hparams.covariates_preprocess, dtype="past_covs")
        past_cov_pipe.fit_or_load(data.X_train_past, model_dir, stage=stage)
        data.X_past = past_cov_pipe.transform(data.X_past)
    else:
        past_cov_pipe = None

    # fit and transform future cov data
    if data.future_cov_cols:
        future_cov_pipe = TSPipeline(hparams.covariates_preprocess, dtype="future_covs")
        future_cov_pipe.fit_or_load(data.X_train_future, model_dir, stage=stage)
        data.X_future = future_cov_pipe.transform(data.X_future)
    else:
        future_cov_pipe = None

    # remove low correlation data after preprocessing pipeline
    # THIS is SUPER sketchy as it is not recorded in pipeline !!!
    data.trim_low_corr_covariates(hparams.pca_corr_cutoff)

    # change to DARTS format
    data.to_timeseries()

    # Prediction setup - forecast_horizon
    n_predict = len(data.X) - len(data.y)
    if data.past_cov_cols:
        n_predict += hparams.model_params["output_chunk_length"]

    print(f"Predicting {n_predict} next months.")
    hparams.forecast_horizon = n_predict

    # logging and printing
    configs.fill_train_test_split_info(data)
    print(hparams)

    pipes = (target_pipe, past_cov_pipe, future_cov_pipe)
    return configs, data, pipes


def setup_model(configs: UserConfig, data: TSData) -> tuple[ForecastingModel, int]:
    hparams = configs.hparams
    model_dir = Path(hparams.model_dir)
    train_data, test_data = data.give_target_data(
        stage=hparams.stage, test_set_at_beginning=hparams.test_set_at_beginning
    )

    # =================== instantiate model OR load saved model for further training =============
    # reproducibility - will be used if re-training model after hyper-parameter optimization
    if hparams.manual_seed:
        torch.manual_seed(hparams.manual_seed)

    model_class = getattr(darts.models, hparams.model)
    hparams.model_category = determine_model_category(hparams.model)

    existing_model_path = model_dir.joinpath(hparams.model_name)
    if existing_model_path.exists() and hparams.stage != "train":
        print("Loading pre-trained model:", hparams.model_name)
        # very hacky way to do retraining. Probably should use load_weighs and start new training.
        # model = model_class.load_weights_from_checkpoint(hparams.model_name, best=False)
        model = model_class.load_from_checkpoint(
            hparams.model_name, work_dir=hparams.darts_logs_dir, best=False
        )
        model = train_model(model, train_data, test_data, data, epochs=hparams.trained_epochs + 1)
        for cb in model.trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                cb.wait_count = 0
                break
        # this is arbitrary, retraining for 10 additional epochs
        add_epochs = 10
    else:
        print(f"Setting up new {hparams.model} with name {hparams.model_name}")
        # define model class from parameters
        model_params = prepare_model_params(hparams, data, hparams.device)
        model = model_class(**model_params)
        add_epochs = 0

    return model, add_epochs


def train_model(
    model: ForecastingModel,
    # series: TimeSeries,
    # val_series: TimeSeries,
    data: TSData,
    params: ConfigSchema,
    epochs: int = 0,
) -> ForecastingModel:
    series, val_series = data.give_target_data(
        stage=params.stage, test_set_at_beginning=params.test_set_at_beginning
    )
    if isinstance(model, darts.models.forecasting.auto_arima.AutoARIMA):
        model.fit(series=series)
    else:
        model.fit(
            series=series,
            past_covariates=data.X_past_ts,
            future_covariates=data.X_future_ts,
            val_series=val_series,
            val_past_covariates=data.X_past_ts,
            val_future_covariates=data.X_future_ts,
            epochs=epochs,
        )
    return model


def forecast(model: ForecastingModel, data: TSData, configs: UserConfig) -> TimeSeries:
    hparams = configs.hparams
    stage = hparams.stage
    train_data, test_data = data.give_target_data(
        stage=hparams.stage, test_set_at_beginning=hparams.test_set_at_beginning
    )
    if (
        hparams.model_category == "deep learning"
        and hparams.reload_best_model_for_inference
        and stage != "forecast"
    ):
        print("Reloading best model from training.")
        model = model.load_from_checkpoint(hparams.model_name, work_dir=hparams.darts_logs_dir)

    # Prediction setup - input_series
    if hparams.test_set_at_beginning and stage != "forecast":
        input_series = test_data[: hparams.model_params["input_chunk_length"]]
    else:
        input_series = train_data

    dl_args = {}
    if hparams.model_category == "deep learning":
        dl_args["n_jobs"] = hparams.n_jobs

    if hparams.model_category == "statistical":
        y_hat = model.predict(n=hparams.forecast_horizon)
    else:
        y_hat = model.predict(
            n=hparams.forecast_horizon,
            series=input_series,
            past_covariates=data.X_past_ts,
            future_covariates=data.X_future_ts,
            num_samples=hparams.num_samples,
            **dl_args,
        )

    return y_hat


def infer(configs: UserConfig, data: TSData, target_pipe: TSPipeline):
    hparams = configs.hparams
    hparams.stage = "forecast"
    model_dir = Path(hparams.model_dir)

    model_class = getattr(darts.models, hparams.model)
    if hparams.device == "cpu":
        model = model_class.load_from_checkpoint(
            hparams.model_name, work_dir=hparams.darts_logs_dir, map_location="cpu"
        )
        model.to_cpu()
    else:
        torch.set_float32_matmul_precision("high")
        model = model_class.load_from_checkpoint(
            hparams.model_name, work_dir=hparams.darts_logs_dir
        )

    # Predict and plot the results
    y_hat = forecast(model, data, configs)

    # plot
    y_hat = target_pipe.inverse_transform(y_hat)
    title = f"Forecast: {y_hat.index[0].strftime('%m/%Y')} - {y_hat.index[-1].strftime('%m/%Y')}"
    output_path = model_dir.joinpath("forecast.png")
    fig = plot_timeseries(
        data.y, y_hat, quantiles=(0.01, 0.99), title=title, save_path=output_path
    )
    fig.show()

    # save forecast
    data.y.to_csv(model_dir.joinpath("target.csv"))
    y_hat.to_csv(model_dir.joinpath("target_pred.csv"))

    return y_hat


def save_artifacts(configs: UserConfig, data: TSData, pipes: tuple[TSPipeline]) -> None:
    # save artifacts
    model_dir = Path(configs.hparams.model_dir)
    target_pipe, past_cov_pipe, future_cov_pipe = pipes
    target_pipe.save(model_dir)
    if past_cov_pipe is not None:
        past_cov_pipe.save(model_dir)
    if future_cov_pipe is not None:
        future_cov_pipe.save(model_dir)
    data.save_data_space(model_dir)
    config_file = model_dir.joinpath(f"config_{configs.hparams.model_name}.yaml")
    configs.save_copy(config_file)
    print("Saved model and config file to disk.")


# #########################    MAIN SCRIPT   ###############################
@click.command(
    context_settings=dict(ignore_unknown_options=True),
    help="Pass in your config file. Any config variable can be overwritten via argument passing. "
    "E.g. python train.py --config '/path/to/config' --learning_rate=0.005 --batch_size=128",
)
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
@click.argument("other_args", nargs=-1, type=click.UNPROCESSED)
def main(config: str, cpu: bool, other_args: tuple) -> None:
    """Main training function of delphi - tracked with MLFlow

    Args:
        config (str): Path to user config.yaml file.
        other_args (tuple): Tuple of (--key=value, --key=value, ...) to overwrite config params.
    """
    # parse config from CLI and overwrite any passed parameter
    configs = UserConfig(config)
    kwargs = dict((arg[2:].split("=") for arg in other_args))
    configs.overwrite_hparams_from_dict(kwargs)
    configs.create_model_name()

    # Determine device -- may use cpu even though gpu available
    configs.hparams.device = determine_device() if not cpu else "cpu"

    # configure data, pipelines and model
    configs, data, pipes = setup_data(configs)
    model_dir = Path(configs.hparams.model_dir)
    target_pipe, _, _ = pipes
    model, add_epochs = setup_model(configs, data)
    hparams = configs.hparams

    # mlflow setup
    ml_runs_dir = Path(configs.hparams.project_dir).joinpath("mlruns")
    mlflow.set_tracking_uri("file://" + str(ml_runs_dir))
    mlflow.set_experiment(data.target_name)
    mlflow.pytorch.autolog(log_models=False)
    with mlflow.start_run(description=configs.hparams.model, run_name=configs.hparams.model_name):
        # train
        train_model(model, data, hparams, epochs=add_epochs)
        hparams.trained_epochs = getattr(model, "epochs_trained", None)
        # predict
        y_hat = forecast(model, data, configs)
        # loss calculation
        loss = calculate_loss(data.y_test_ts, y_hat, hparams.metric, n_jobs=hparams.n_jobs)
        hparams.metrics[f"{hparams.stage}_{hparams.metric}"] = float(loss)
        # plot test forecast
        title = f"{hparams.stage}_{hparams.metric} = {loss:.2f}%"
        output_path = model_dir.joinpath(f"{title.split(' = ')[0]}.png")
        y_hat = target_pipe.inverse_transform(y_hat)
        fig = plot_timeseries(
            data.y, y_hat, quantiles=(0.01, 0.99), title=title, save_path=output_path
        )
        fig.show()

        # plot real forecast
        if hparams.stage == "forecast" or hparams.test_set_at_beginning:
            y_hat = infer(configs, data, target_pipe)

        # logging
        mlflow.log_metric(f"{hparams.stage}_{hparams.metric}", loss)
        mlflow.log_params(asdict(hparams))
        save_artifacts(configs, data, pipes)
        mlflow.log_artifacts(model_dir)


if __name__ == "__main__":
    main()
