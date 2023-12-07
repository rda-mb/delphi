"""
This module provides classes for loading, preprocessing, and transforming time series data for
forecasting purposes. It includes classes for loading raw data from different file formats,
preprocessing the data, splitting it into train/test/validation sets, and
handling time series specific transformations. Past and future covariates are also separated and
preprocessed separately as PCA would not be possible otherwise.

Classes:
    DataLoader: A class to load raw data from various file formats and return one DataFrame.
    TSData: A class to handle time series data, splitting:
        - target, past covariates and future covariates,
        - train, test and validation sets,
        and converting it to Darts TimeSeries format.
    TSPipeline: A class to build and apply preprocessing pipelines for time series data,
        including target and covariate transformations.
"""
from datetime import datetime, timedelta
from pathlib import Path

from darts import TimeSeries
from darts.timeseries import concatenate
import joblib
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import sklearn.preprocessing

from delphi.utils.pandas import trim_initial_zeros, trim_low_correlation_variables


class TSLoader:
    """
    A class to load raw data from various file formats, check data validity and build a
    single DataFrame.

    Attributes:
        params (object): An object containing user-defined parameters for loading and
        preprocessing data.

    Methods:
        load_data: Load raw data from specified file format and preprocess it.
        find_target_variable: Find the target variable column in the dataset.
        crop_date_range: Crop the dataset to the specified date range.
        trim_initial_zeros: Trim initial zeros from the dataset.
    """

    def __init__(self, data_sources: list[str]):
        self.data_sources = data_sources
        self.df = pd.DataFrame()
        self.target_column = ""
        self.load_data()

    def load_data(self) -> None:
        """Load the data from the specified data sources. Supported file formats
        are .csv and .parquet so far. Data sources are passed as a list in config.yaml
        ```yaml
        data_sources:
          - /path/to/data1
          - /path/to/data2
        ```
        Pre-requisite: Data needs to contain a Date column and if several data sources are
        loaded they need to have matching sampling frequency: 'M', 'MS', 'Y', etc..

        All data are joined together in one pandas DataFrame.
        Optionnaly, trailing zeros at the begining of the target series are removed.
        Optionnaly, covariates with low correlation to the target series are removed.

        Raises:
            AttributeError: If the loaded data does not contain a 'Date' column.
            ImportError: If the data source file format is not .parquet or .csv.
        """
        # load all data as input in user params
        df = pd.DataFrame()
        for d in self.data_sources:
            if d.endswith(".parquet"):
                print("Loading data from:", d)
                curr_df = pd.read_parquet(d)

            elif d.endswith(".csv"):
                print("Loading data from:", d)
                try:
                    curr_df = pd.read_csv(d, parse_dates=True, index_col="Date")
                except Exception:
                    raise AttributeError(
                        "Loaded data should contain one 'Date' column."
                        "Use df = df.set_index('your_date')"
                        "df.index.name = 'Date'"
                        "df.to_csv(pathname)"
                    )
            else:
                raise ImportError("Load data from .parquet or .csv format.")

            if df.empty:
                # first data source -> copying data to df
                df = curr_df
                freq = pd.infer_freq(df.index)
                continue

            # check if data freq matches
            if pd.infer_freq(curr_df.index) != freq:
                raise AttributeError(
                    "Loaded data should have matching sampling frequency."
                    f"Found {freq} and {pd.infer_freq(curr_df.index)}"
                )
            # other data sources -> join to df if freq matches
            df = df.join(curr_df, how="outer")

        self.df = df

    def crop_date_range(self, min_date: str = "min", max_date: str = "max") -> pd.DataFrame:
        """Crop the date range of the DataFrame according to user params.

        Args:
            df (pd.DataFrame): The input DataFrame with a datetime index.

        Returns:
            pd.DataFrame: The cropped DataFrame.
        """
        # crop requested dates
        if min_date == "min":
            min_date = self.df.index[0]
        if max_date == "today":
            # this will effectively stop the use of future covariates
            last_month = datetime.today() - timedelta(days=1)
            max_date = last_month.strftime("%Y-%m-%d")
        elif max_date == "max":
            max_date = self.df.index[-1]

        self.df = self.df[min_date:max_date]

    def trim_data(
        self, trim_trailing_zeros: bool = False, low_corr_trim_thresh: float = 0.0
    ) -> pd.DataFrame:
        """Trim the data according to user config.

        Args:
            trim_initial_zeros (bool, optional): Whether to trim initial trailing zeros.
                Defaults to False.
            trim_low_corr_variables (float, optional): Correlation threshold used to discard
                variables weakly correlated to target variable. Defaults to 0.0.

        Returns:
            DataFrame: Final Dataframe.
        """
        # discard initial trailing zeros
        if trim_trailing_zeros:
            self.df = trim_initial_zeros(self.df, self.target_column)
        # discard low correlation variables data
        self.df = trim_low_correlation_variables(
            self.df, self.df[self.target_column], thresh=low_corr_trim_thresh
        )
        return self.df

    def validate_target_variable(self, target: str | list[str]) -> str:
        """Find the target column based on user params.

        Returns:
            str: The name of the target column in the dataset.

        Raises:
            AttributeError: If there is no match or more than one match for the target column
            in the dataset.
        """
        # find target column based on user params
        df_with_col = self.df.copy()
        if not isinstance(target, list):
            target = [target]
        for word in target:
            df_with_col = df_with_col.loc[:, df_with_col.columns.str.contains(word)]
        if len(df_with_col.columns) == 0:
            raise AttributeError(
                "Target name has no match in the dataset. Check your data and config names. "
                f"Requested target column with {' '.join(target)}. "
                f"Found None"
            )
        elif len(df_with_col.columns) > 1:
            raise AttributeError(
                "Target name has more than one match in the dataset. "
                "Use additional 'categories' in config.target. "
                f"Requested target column with {' '.join(target)}. "
                f"Found {len(df_with_col.columns)} matches: {' '.join(df_with_col.columns)}."
            )

        self.target_column = df_with_col.columns[0]
        return self.target_column


class TSData:
    """
    A class to handle time series data, splitting target, past covariates and future covariates,
    then splitting it into training, validation, and testing sets, and converting it to
    Darts TimeSeries format.

    Attributes:
        data (pd.DataFrame): The input time series data as a DataFrame.
        target (str): Column name in dataframe data to use as target.
        stage (str): Either of 'train' or 'forecast'. Default: 'train'.

    Methods:
        split_target_from_covariates: Split the target variable from the covariates.
        split_past_future_covariates: Split past and future covariates.
        split_train_test: Split the data into training, validation, and testing sets.
        save_covariates: Save information about the used covariates.
        load_covariates: Load information about the used covariates.
        trim_low_corr_covariates: Utility funcion to trim low correlation covariates.
        to_timeseries: Convert the data to Darts TimeSeries format.
    """

    def __init__(self, data: pd.DataFrame, target: str, stage: str = "train"):
        self.raw_data = data
        self.target_name = target
        self.stage = stage
        self.past_cov_cols = []
        self.future_cov_cols = []
        self.y_hat = None

    def split_target_from_covariates(self) -> None:
        """Split the target column from the covariates in the DataFrame self.raw_data."""
        self.X = self.raw_data.drop(self.target_name, axis=1)
        self.y = self.raw_data[[self.target_name]]

    def split_past_future_covariates(self, force_use_only_past_covs: bool = False) -> None:
        """Split the past and future covariates in the DataFrame self.X.
        1- find the last non null value in self.y and the date associated
            -> this defines last_target_date
        2- find in X columns which have finite values after the last_target_date
            -> this defines the future covariates
        3 - All other are past covariates.
        """

        # find the last non null value in y and the date associated
        last_target_date = self.y.dropna().index[-1] + timedelta(days=1)
        self.y = self.y[:last_target_date]

        # find in X columns which have non null values after the last_target_date
        if self.X[last_target_date:].empty:
            self.future_cov_cols = []
            self.past_cov_cols = list(self.X.columns)
        else:
            self.future_cov_cols = list(
                self.X[last_target_date:].dropna(axis=1, how="any").columns
            )
            self.past_cov_cols = [c for c in self.X.columns if c not in self.future_cov_cols]

        # scenario in which we want to use future know covariates as past covariates
        if force_use_only_past_covs and self.future_cov_cols:
            self.future_cov_cols, self.past_cov_cols = self.past_cov_cols, self.future_cov_cols

        # use notation X for covariates
        if self.past_cov_cols:
            self.X_past = self.X[self.past_cov_cols]
            # augment artificially X_past data into the future, needed for future preds
        else:
            self.X_past = None

        if self.future_cov_cols:
            self.X_future = self.X[self.future_cov_cols]
        else:
            self.X_future = None

    def split_train_test(self, split: float, test_set_at_beginning: bool = False) -> None:
        """Split the data into train and test sets according to the given split ratio.
        The test set is further split as: 2/3 test and 1/3 validation. With Darts, only y needs
        splitting, the covariates X are split automatically when model.fit() is called.

        Args:
            split (float): The split ratio for train/test sets.
            use_val_set (bool): Whether to have train/test/split sets. Default False.
            test_set_at_beginning (bool): Select test set at the beginning of the timeseries.
                Default uses test set in the end of the timeseries.
        """

        if test_set_at_beginning:
            # case test set in the beginning of the data set
            split_date = self.y.reset_index()["Date"].quantile(q=1 - split)
            self.y_test = self.y[:split_date]
            self.y_train = self.y[split_date + timedelta(seconds=1) :]
        else:
            # case test set in the end of the data set
            split_date = self.y.reset_index()["Date"].quantile(q=split)
            self.y_train = self.y[:split_date]
            self.y_test = self.y[split_date + timedelta(seconds=1) :]

        # training covariates needed only for pipeline transforms
        if self.past_cov_cols:
            self.X_train_past = self.X_past.loc[self.y_train.index, :]
        if self.future_cov_cols:
            self.X_train_future = self.X_future.loc[self.y_train.index, :]

    def give_target_data(
        self, stage: str = "train", test_set_at_beginning: bool = False
    ) -> tuple[TimeSeries, TimeSeries]:
        """Gives the training target set and validation target set based on stage.

        Args:
            stage (str, optional): Stage of the model training. Defaults to "train".

        Returns:
            tuple[TimeSeries, TimeSeries]: Train and validation TimeSeries suitable for the stage.
        """
        if stage == "train":
            train_data = self.y_train_ts
        elif stage == "forecast":
            if test_set_at_beginning:
                train_data = concatenate([self.y_test_ts, self.y_train_ts])
            else:
                train_data = concatenate([self.y_train_ts, self.y_test_ts])
        return train_data, self.y_test_ts

    def save_data_space(self, path: Path) -> None:
        """Save the target and covariates used to produce the model to a yaml file.

        Args:
            path (Path): The directory path where the yaml file is saved.
        """
        # save info on covariates used
        output = {
            "targets": [self.target_name],
            "past_covariates": self.past_cov_cols,
            "future_covariates": self.future_cov_cols,
        }

        with open(path.joinpath("data_space.yaml"), "w") as file:
            yaml.dump(output, file)

    def check_data_space_consistency(self, path: Path) -> None:
        """Load the used covariates/target from a yaml file and checks data consistency,
        i.e. that the same columns are reused from when a model was first trained.

        Args:
            path (Path): Model directory containing data_space.yaml.
        """
        with open(path.joinpath("data_space.yaml")) as file:
            data_space = yaml.safe_load(file)

        X_past_num_cols = len(self.X_past.columns) if self.past_cov_cols else 0
        X_future_num_cols = len(self.X_future.columns) if self.future_cov_cols else 0
        if len(data_space["past_covariates"]) != X_past_num_cols:
            raise ImportError(
                f"Model originally trained with {len(data_space['past_covariates'])}."
                f"Found {X_past_num_cols} in current data."
            )
        if len(data_space["future_covariates"]) != X_future_num_cols:
            raise ImportError(
                f"Model originally trained with {len(data_space['future_covariates'])}."
                f"Found {X_future_num_cols} in current data."
            )

    def trim_low_corr_covariates(self, corr_cutoff: float) -> None:
        """Trim low correlation covariates according to the given correlation cutoff.

        Args:
            corr_cutoff (float): The correlation cutoff value.
        """
        # whether to keep all pca components or remove low correlation components
        if self.past_cov_cols:
            self.X_past = trim_low_correlation_variables(
                self.X_past, self.y_train[self.target_name], corr_cutoff
            )
        if self.future_cov_cols:
            self.X_future = trim_low_correlation_variables(
                self.X_future, self.y_train[self.target_name], corr_cutoff
            )

    def to_timeseries(self):
        """Convert from pandas DataFrame to Darts TimeSeries object.

        Covariates and target series are first preprocessed with scikit-learn
        so this .to_timeseries() method is called in the very end of data preprocessing
        after fit_transform.

        Future upgrade: setup like this currently because Darts does not have PCA
        If PCA becomes available, or we write our own PCA() class compatible with
        Darts's TimeSeries then we could simplify a lot of the code.

        However, let's keep in mind that switching to TimeSeries as a very last step
        may be a better approach as we could also use other libraries e.g. PyTorch Forecasting.
        """
        # targets
        self.y_train_ts = TimeSeries.from_series(self.y_train).astype(np.float32)
        self.y_test_ts = TimeSeries.from_series(self.y_test).astype(np.float32)
        # past covariates
        if self.past_cov_cols:
            self.X_past_ts = TimeSeries.from_dataframe(self.X_past).astype(np.float32)
        else:
            self.X_past_ts = None
        # future covariates
        if self.future_cov_cols:
            self.X_future_ts = TimeSeries.from_dataframe(self.X_future).astype(np.float32)
        else:
            self.X_future_ts = None

    def save(self, variable: str, path: Path, output_name: str = "target") -> None:
        """Save data from TSData instance to disk.

        Args:
            variable (str): TSData variable timeseries.
            path (Path): Path to save the data to.
            output_name (str, optional): Name to save variable as. Defaults to "target".
        """
        series = getattr(self, variable, None)
        path = path.joinpath(output_name + ".parquet")
        series.pd_dataframe().to_parquet(path)


class TSPipeline:
    """
    A class to build and apply preprocessing pipelines for time series data, including target and
    covariate transformations. Currently. the pipelines are built with Scikit-learn.

    Attributes:
        preprocess_steps (dict): User defined steps to apply to data. Fetched from config file.
        dtype (str): Either of 'target', 'past_covs' or future 'covs'
        stage (str): The stage for which the pipeline is being built ('train' or 'forecast').

    Methods:
        build: Build the preprocessing pipelines for target and covariate transformations.
        fit: Fit the preprocessing pipelines on the training data.
        transform: Transform the time series data using the fitted preprocessing pipelines.
        inverse_transform: Inverse transform the predictions to their original scale.
        save: Save the preprocessing pipelines to a joblib file.
        load: Load the preprocessing pipelines from a joblib file.
    """

    def __init__(self, preprocess_steps: dict, dtype: str):
        self.preprocess_steps = preprocess_steps
        self.type = dtype

    def build(self) -> None:
        """Build the pipelines for preprocessing past covariates,
        future covariates, or target.
        """

        steps = []
        for n, (fn, kwargs) in enumerate(self.preprocess_steps.items()):
            fn = PCA if fn == "PCA" else getattr(sklearn.preprocessing, fn)
            fn_call = fn(**kwargs) if kwargs else fn()
            steps.append((f"{self.type}_step_{n}", fn_call))
        self.pipe = Pipeline(steps)

    def fit_or_load(self, df: pd.DataFrame, model_path: str | Path, stage: str = "train") -> None:
        """Fit the pipelines on the training dataset.

        Args:
            df (pd.DataFrame): The time series dataframe containing the training data.
        """
        # keeping track of original column names
        self.raw_columns = df.columns
        if stage == "train":
            self.build()
            self.pipe.fit(df.values)
        else:
            self.load(model_path)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the time series data using the fitted preprocessing pipelines.
        DataFrame -> np.ndarray -> transform -> np.ndarray -> pd.DataFrame

        Args:
            df (pd.DataFrame): The input time series dataframe to be transformed.

        Returns:
            pd.DataFrame: The transformed dataframes.
        """
        # TRANSFORM target, train/test
        transformed_df = self.pipe.transform(df.values)
        if transformed_df.shape == df.shape:
            transformed_df = pd.DataFrame(transformed_df, index=df.index, columns=df.columns)
        else:
            transformed_df = pd.DataFrame(transformed_df, index=df.index)

        return transformed_df

    def inverse_transform(self, df: TimeSeries | pd.DataFrame) -> pd.DataFrame:
        """Inverse transform the predictions to their original scale. Targets with many samples
        (probabilistic forecast) are inverted column-wise. Covariates are simply inverted.

        Args:
            df (TimeSeries | DataFrame): Input predictions to be inverse transformed.

        Returns:
            DataFrame: The inverse transformed dataframe with same shape as input.
        """

        if isinstance(df, TimeSeries):
            df = df.pd_dataframe()
        # inverse transform all samples
        samples = self.pipe.inverse_transform(df.values)
        columns = self.raw_columns if self.type != "target" else None
        return pd.DataFrame(samples, index=df.index, columns=columns)
        # if self.type == "target":
        #     # mean prediction and quantiles
        #     pred_mean = np.mean(samples, axis=1)
        #     pred_q1 = np.quantile(samples, q=0.01, axis=1)
        #     pred_q99 = np.quantile(samples, q=0.99, axis=1)
        #     # rebuild dataframe
        #     df_detransformed = pd.DataFrame(pred_mean, index=df.index, columns=["pred"])
        #     df_detransformed["low_q"] = pred_q1
        #     df_detransformed["high_q"] = pred_q99
        #     return df_detransformed
        # else:
        #     return pd.DataFrame(samples, index=df.index, columns=self.raw_columns)

    def save(self, path: Path) -> None:
        """
        Save the TSPipeline object to a joblib file.

        Args:
            path (Path): The directory where the 'data_pipeline.joblib' file is saved.
        """
        pipe_file = path.joinpath(f"{self.type}_pipeline.joblib")
        joblib.dump(self.pipe, pipe_file)

    def load(self, path: str | Path) -> None:
        """Loads a TSPipeline object from a joblib file.
        Args:
            path (Path): The directory containing the 'data_pipeline.joblib' file.
            Default None, will be loaded from same folder as config file.

        Returns:
            DataPipeline: The loaded DataPipeline object.

        Raises:
            FileNotFoundError: If the 'data_pipeline.joblib' file is not found in the specified
            path.
        """
        if isinstance(path, str):
            path = Path(path)
        pipe_file = path.joinpath(f"{self.type}_pipeline.joblib")
        if not pipe_file.exists():
            raise FileNotFoundError(
                f"File '{self.type}_pipeline.joblib' not found in the specified path: {path}"
            )

        # Update the attributes of the current object
        loaded_pipeline = joblib.load(pipe_file)
        self.pipe = loaded_pipeline
