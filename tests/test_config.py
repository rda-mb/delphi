""" Test functions for UserConfig and ConfigSchema classes"""
from pathlib import Path

# import os
import shutil
import tempfile

import pytest
import yaml

from delphi.core.config import ConfigSchema, UserConfig
from delphi.utils.helpers import is_path


# Dummy config data for testing
dummy_config = """
project_name: Test Project
version_name: v1
project_dir: /path/to/project
data_sources:
- data_source1.csv
- data_source2.csv
target:
- target
model: ModelName
model_params:
  param1: 1
  param2: 0.5
  param3: string_value
  param4: true
trainer_params:
  n_epochs: 100
  batch_size: 64
data_date_start: 2000-01-01  # use 'min' or a date
data_date_stop: 2024-06-01   # use 'max' or 'today' or a date
trim_initial_zeros: true
trim_low_corr_variables_upfront: false
split: .85        # train/test %
covariates_preprocess:
  PCA:
    n_components: 0.95
pca_corr_cutoff: 0
"""


@pytest.fixture
def config_file():
    test_dir = tempfile.mkdtemp()
    config_file = Path(test_dir, "config.yaml")
    with open(config_file, "w") as f:
        f.write(dummy_config)
    yield config_file
    shutil.rmtree(test_dir)


def test_is_path(config_file):
    assert isinstance(is_path(config_file), Path)
    with pytest.raises(FileNotFoundError):
        is_path("/path/that/does/not/exist")


def test_ConfigSchema():
    raw_config = yaml.safe_load(dummy_config)
    config = ConfigSchema(**raw_config)
    assert config.project_name == "Test Project"
    assert config.version_name == "v1"


def test_UserConfig(config_file):
    user_config = UserConfig(config_file)
    assert isinstance(user_config, UserConfig)
    assert isinstance(user_config.hparams, ConfigSchema)


def test_fill_additional_info(config_file):
    user_config = UserConfig(config_file)
    user_config.fill_basic_info()
    assert hasattr(user_config.hparams, "build_date")
    assert hasattr(user_config.hparams, "delphi_version")


# Add more test functions for other methods in the UserConfig class
