"""
This module contains utility functions and classes for loading and validating user configurations
for the Delphi time-series forecasting model.

Classes:
ConfigSchema (dataclass): A dataclass for validating and storing user configuration data.
UserConfig: A class for loading, validating, storing and saving user config from a YAML file.

Functions:
is_path(path: str) -> Path: Checks and returns the absolute path of a given file.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import os
from pathlib import Path
import re
import socket

import darts
import optuna
import pytorch_lightning as pl
import ruamel.yaml as yaml
import torch

from delphi.core.data import TSData
from delphi.utils.helpers import get_git_commit_hash, is_path
from delphi.version import __version__


@dataclass
class ConfigSchema:
    """
    A dataclass for validating and storing user configuration data.
    """

    # variable_name: type = default_value
    project_name: str
    version_name: str
    project_dir: str
    data_sources: list[str]
    target: str | list[str]
    model: str
    model_params: dict[str, int | float | str | bool]

    # parameters with default
    trainer_params: dict[str, int | float | str | bool] = field(default_factory=dict)
    stage: str = "train"
    metric: str = "mape"
    trim_initial_zeros: bool = False
    trim_low_corr_variables_upfront: float = 0.0
    force_use_only_past_covs: bool = False
    split: float = 0.85  # train/test %
    test_set_at_beginning: bool = False
    pca_corr_cutoff: float = 0.0
    forecast_horizon: int = 60
    data_date_start: str = "min"
    data_date_stop: str = "max"
    covariates_preprocess: dict[str, dict[str, int | float | list]] = field(default_factory=dict)
    target_preprocess: dict[str, dict[str, int | float | list]] = field(default_factory=dict)
    reload_best_model_for_inference: bool = True
    num_samples: int = 1
    n_jobs: int = 8
    manual_seed: int = None

    # parameters saved automatically -- need to be in schema for loading saved models
    model_name: str = None
    model_category: str = None
    build_date: str = None
    config_file: str = None
    # darts_logs: str = None
    hostname: str = None
    delphi_version: str = None
    darts_version: str = None
    pytorch_version: str = None
    pytorch_lightning_version: str = None
    optuna_version: str = None
    git_commit_hash: str = None
    raw_data_info: dict[str, str | bool | tuple] = field(default_factory=dict)
    training_data_info: dict[str, str | bool | tuple] = field(default_factory=dict)
    test_data_info: dict[str, str | bool | tuple] = field(default_factory=dict)
    val_data_info: dict[str, str | bool | tuple] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    device: str = None
    trained_epochs: int = None
    model_dir: str = None
    darts_logs_dir: str = None

    def __str__(self):
        """
        Pretty-prints the configuration dictionary, showing key-value pairs and comments.
        """
        result = []
        result.append(
            "\n================================ USER CONFIGS =================================="
        )
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                result.append(k)
                for kp, vp in v.items():
                    if isinstance(vp, dict):
                        result.append(f" - {kp}")
                        for kpp, vpp in vp.items():
                            result.append(f"   - {kpp.ljust(35)} {vpp}")
                    else:
                        result.append(f" - {kp.ljust(37)} {vp}")
            else:
                result.append(f"{k.ljust(40)} {v}")
        result.append(
            "================================================================================\n"
        )
        return "\n".join(result)


class UserConfig:
    """
    A class for loading, validating, and storing user configuration from a YAML file.

    Attributes:
        config_file (Path): The path to the user configuration file.
        original_config (ruamel.yaml.comments.CommentedMap): The original configuration dictionary.
        hparams (ConfigSchema): A ConfigSchema instance containing the validated user config.

    Methods:
        fill_additional_info: Adds additional information to the configuration dictionary.
        add_train_test_split_info(train, test): Adds train/test split info to the config.
        create_model_name: Creates a model name based on the configuration.
        save_post_training_info(y_true, y_base,y_hat, device, epochs, model_path):
            Saves post-training information to the configuration dictionary.
        save_copy(output_dir): Saves a copy of the configuration dictionary to a file.
        pprint: Pretty-prints the configuration dictionary.
    """

    def __init__(self, config_file: str):
        self.config_file = is_path(config_file)
        print(f"Loading user config from: {config_file}")

        # load user config file
        with open(self.config_file) as cf:
            raw_config = yaml.YAML().load(cf)

        # store the original configuration dictionary
        self.original_config = raw_config

        # validate and store configuration using dataclass
        self.hparams = ConfigSchema(**raw_config)

        if "build_date" not in raw_config:
            self.fill_basic_info()

    def overwrite_hparams_from_dict(self, args: dict) -> None:
        # change tuple to dict (key, value)
        for k, v in args.items():
            # check in top level variables
            if hasattr(self.hparams, k):
                old_value = getattr(self.hparams, k)
                type_ = type(old_value) if old_value else str
                setattr(self.hparams, k, type_(v))
                print(f"Setting config variable {k}={v}")
            # check in model_params
            elif k in self.hparams.model_params:
                old_value = self.hparams.model_params[k]
                type_ = type(old_value) if old_value else str
                self.hparams.model_params[k] = type_(v)
                print(f"Setting model variable {k}={v}")
            # check in trainer_params
            elif k in self.hparams.trainer_params:
                old_value = self.hparams.trainer_params[k]
                type_ = type(old_value) if old_value else str
                self.hparams.trainer_params[k] = type_(v)
                print(f"Setting trainer variable {k}={v}")
            else:
                print(
                    f"WARNING: passed argument '{k}' not found in user config. "
                    "Config variables not changed. If trying to overwrite variable through CLI "
                    "use --{variable}={new_value} syntax."
                )

    def fill_basic_info(self):
        """
        Adds additional information to the configuration dictionary, such as build date,
        config file path, hostname, and library versions.
        """
        self.hparams.build_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.hparams.config_file = str(self.config_file)
        # self.hparams.darts_logs = str(Path(os.getcwd()).joinpath("darts_logs"))
        self.hparams.hostname = socket.gethostname()
        self.hparams.darts_version = str(darts.__version__)
        self.hparams.pytorch_version = str(torch.__version__)
        self.hparams.pytorch_lightning_version = str(pl.__version__)
        self.hparams.optuna_version = str(optuna.__version__)
        self.hparams.delphi_version = str(__version__)
        self.hparams.git_commit_hash = get_git_commit_hash()

    def fill_train_test_split_info(self, data: TSData):
        """
        Adds train/test split information to the configuration dictionary,
        including start and end dates and durations for training and testing data.

        Args:
            train (TimeSeries): The training time series.
            test (TimeSeries): The testing time series.
        """
        # all split datasets
        y_train = data.y_train_ts
        y_test = data.y_test_ts
        past_covs = data.X_past_ts
        future_covs = data.X_future_ts
        # number of features
        n_past_covs_features = past_covs.n_components if data.past_cov_cols else 0
        n_future_covs_features = future_covs.n_components if data.future_cov_cols else 0
        n_features = n_past_covs_features + n_future_covs_features

        # raw data info
        data_info = {
            "shape": (len(data.y), len(data.X.columns)),
            "nb_past_covs": len(data.past_cov_cols),
            "nb_future_covs": len(data.future_cov_cols),
        }
        self.hparams.raw_data_info = data_info

        # training
        training_info = {
            "shape": (y_train.n_timesteps, n_features),
            "start": y_train.start_time().strftime("%Y-%m-%d"),
            "stop": y_train.end_time().strftime("%Y-%m-%d"),
            "duration": f"{y_train.duration.days} days",
        }
        self.hparams.training_data_info = training_info

        # test
        testing_info = {
            "shape": (y_test.n_timesteps, n_features),
            "start": y_test.start_time().strftime("%Y-%m-%d"),
            "stop": y_test.end_time().strftime("%Y-%m-%d"),
            "duration": f"{y_test.duration.days} days",
        }
        self.hparams.test_data_info = testing_info

    def create_model_name(self) -> None:
        """
        Creates a model name based on the configuration and adds it to the config dictionary.

        Returns:
            model_name (str): The generated model name.
        """
        if self.hparams.model_name:
            return

        target = self.hparams.target
        if not isinstance(target, list):
            target = [target]
        sub_model = "_".join(target)
        # remove punctuation if any
        sub_model = re.sub(r"[^A-Za-z0-9]+", "_", sub_model)
        # build unique name
        base_name = f"{self.hparams.model}_{self.hparams.version_name}"
        model_name = f"{base_name}_{sub_model}".replace(" ", "")
        self.hparams.model_name = model_name

    def save_copy(self, output_path: Path):
        """
        Saves a copy of the configuration dictionary to a file.

        Args:
            output_dir (Path): The directory to save the configuration file.
        """
        # overwrite original_config with hparams variables
        for k, v in asdict(self.hparams).items():
            if k not in self.original_config:
                self.original_config[k] = v
            else:
                # only overwrite if value has changed
                if self.original_config[k] != v:
                    self.original_config[k] = v

        # add some comments for readability
        self.original_config.yaml_set_comment_before_after_key("model_name", "")
        self.original_config.yaml_set_comment_before_after_key(
            "model_name", "parameters specific to saved model"
        )
        self.original_config.yaml_set_comment_before_after_key("training_data", "")
        self.original_config.yaml_set_comment_before_after_key(
            "training_data", "Data shape and split info"
        )
        self.original_config.yaml_set_comment_before_after_key("device", "")
        self.original_config.yaml_set_comment_before_after_key(
            "device", "trained model parameters"
        )

        with open(output_path, "w") as file:
            yaml.YAML().dump(self.original_config, file)
