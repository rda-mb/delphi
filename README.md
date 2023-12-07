![build](https://github.com/rda-mb/delphi/actions/workflows/build.yml/badge.svg)
![Supported versions](https://img.shields.io/badge/python-3.10-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# 🔮 Delphi
This module, named after the [Oracle of Delphi](https://en.wikipedia.org/wiki/Pythia) to signify its ability to make accurate predictions 😉, builds upon the DARTS library for TimeSeries forecasting. The training of models revolves around a `yaml` configuration file, which contains all the parameters used to initialize and train a model. Once trained, the configuration file and data preprocessing pipeline are saved, creating a self-contained system for future inference. With this module, any model from the DARTS library can be easily used by specifying its name in the configuration file, along with the necessary input parameters. This module makes use of one central classe `UserConfig` to handle user configurations and save a unique copy after a model is trained to ensure reproducibility. The idea is that the code does not need editing as every parameters that can possibly vary is changed in the config file.

## 🚀 Capabilities
- data preprocessing with sklearn pipeline: imputation, normalization, standardization, PCA, cutoff of variables with low correlations to target variable, etc ...
- hyperparameter tuning of models on training dataset with [optuna](https://github.com/optuna/optuna)
- model training on training set to select final model - ** model performance tracking TBD **
- model retraining including train + test sets to evaluate model performance
- final model retraining up to present date + forecasting
- historical forecasting for reference
- backtesting for performance evaluation
- integration to the database?

## 🧱 Installation
The repo is setup so as to install `delphi` as a package and follows the new pyproject.toml file convention. After cloning the repo and creating a virtual environment, pip install the repo in editable mode (-e). Editable mode means that a symbolic link is created between the installed package and the original source code on your local machine. Any changes you make to the source code of the package will be immediately reflected in your project, without having to reinstall the package each time you make a change. This facilitates unit testing as well as calling functions and classes accross the different files in the repo. Finally there is no direct way to install PyTorch with CUDA support via the settings file, so it needs to be installed prior if setting up your environment on the Data Analyst Work Station.

```console
$ git clone https://github.com/MaerskBroker/delphi.git
$ cd delphi
```

Then with conda:
```console
$ conda create --name delphi python=3.10
$ conda activate delphi
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ pip install -e .
```

Without conda
```console
$ python -m virtualenv delphi
$ source delphi/bin/activate
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$ pip install -e .
```

## 📜 Pre-requisites
### Data format
Data to be loaded should be time series from `.parquet` or `.csv` file, and should contain:
- One `Date` column with regular frequency (e.g. monthly or yearly).
- One or several variable columns, one of which is the target.

| Date       | variable_1 | variable_2 | ... | variable_n |
| ---------- | ---------- | ---------- | --- | ---------- |
| 2000-01-01 | 24.5       | 0.765      | ... | 2035       |
| 2001-01-01 | 25.3       | 0.782      | ... | 2050       |
| 2002-01-01 | 26.1       | 0.799      | ... | 2065       |
| 2003-01-01 | 26.9       | 0.816      | ... | 2080       |
| 2004-01-01 | 27.7       | 0.833      | ... | 2095       |
| 2005-01-01 | 28.5       | 0.850      | ... | 2110       |
| 2006-01-01 | 29.3       | 0.867      | ... | 2125       |
| 2007-01-01 | 30.1       | 0.884      | ... | 2140       |
| 2008-01-01 | 30.9       | 0.901      | ... | 2155       |
| 2009-01-01 | 31.7       | 0.918      | ... | 2170       |

One or several data sources can be specified in the config file, ony prerequisite is that they have identical frequencies. The data is loaded as follows:
* All data sources are joined together
* `target` column is isolated
* variables that have all non NaN values after the last target value become `future covariates`
* all other variables become `past covariates`

as per Darts definition of these variables.

### Config file
Every run uses a config file in `.yaml` format to pass in user configurations such as model parameters, date range to use etc...
<details>

<summary>Check for an example config.yaml</summary>

```yaml
# main config to train models
project_name: ts_forecast                                 
version_name: v0                     # Make run unique by changing this counter
project_dir: \path\to\folder

# model specific settings
data_sources:
- .\ext_data.parquet
- .\imf.csv
# all defined values here should be contained in target variable name
# useful for organizing category, sub-category, etc...
# can also use the full target name as is
target:
- Price

# model parameters
model: TFTModel
model_params:
  input_chunk_length: 8
  output_chunk_length: 3
  hidden_size: 32
  lstm_layers: 1
  num_attention_heads: 4
  full_attention: true
  feed_forward: GatedResidualNetwork
  dropout: 0.1
  hidden_continuous_size: 32
  add_relative_index: true  # not a model param but somehow input into the model instantiation
  # By default, the TFT model is probabilistic and uses a likelihood instead (QuantileRegression). 
  loss_fn:
  quantiles: [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.975, 0.99]
  likelihood: QuantileRegression

# trainer params for deep learning models (TFT, ...)
trainer_params:
  n_epochs: 500
  batch_size: 32
  learning_rate: 0.001
  optimizer: Adam
  lr_scheduler: ReduceLROnPlateau
  # callbacks
  early_stopping:
    monitor: val_loss
    patience: 15
    min_delta: 0.01
    mode: min
    verbose: true

# data preprocessing parameters
stage: train
data_date_start: 2000-01-01  # use 'min' or a date
data_date_stop: today   # use 'max' or 'today' or a date
forecast_horizon: 60
trim_initial_zeros: true
trim_low_corr_variables_upfront: 0
split: .85        # train/test %
test_set_at_beginning: true
covariates_preprocess:
  MinMaxScaler:
  PCA:
    n_components: 0.95
target_preprocess:
  MaxAbsScaler:
pca_corr_cutoff: 0

# prediction parameters
reload_best_model_for_inference: false
num_samples: 100
n_jobs: 32
```

</details>

## 🧠 Models tested
Check in \library for example of model parameters to use with these models.

Statistical models:
* `AutoARIMA`

Machine learning models:
* TBD (`LinearRegressionModel`, `RandomForest`)

Boosting Models:
* `XGBModel`
* `LightGBMModel`
* `CatBoostModel`

Deep learning models:
* `TFTModel`
* `TransformerModel`
* `TCNModel`
* `RNNModel`
* `BlockRNNModel`

## ▶️ Usage
from your local environment

Optimizing a model:
```console
(.env)$ python hyperopt.py --config ./darts_config.yaml
```
This saves a hyperopt_config.yaml in trained_models/{target_name}/{model_name}

Training a model:
```console
(.env)$ python train.py --config trained_models/{target_name}/{model_name}/hyperopt_config.yaml
```

Making a forecast from a trained model
```console
(.env)$ python infer.py --config trained_models/{target_name}/{model_name}/config_{model_name}.yaml
```

After initial training (`stage = 'train'`), a model can be loaded and further trained with the most recent data (`stage = 'forecast'`). If using the test set at the beginning (`test_set_beginning = True`), retraining the model may not be necessary as the model already knows the latest data.

## ⚠️ Warnings
Work in progress. Let's discuss how this code should be structured to accomodate everyone's need. In particular we should agree on data format.

## 💁 Contributing
Please check the contribution guidelines.
