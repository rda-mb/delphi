# """ Test functions for DataLoader, TSData, TSPipeline classes"""

# # import shutil
# import tempfile

# import pandas as pd
# import pytest

# from delphi.core.config import ConfigSchema

# # import numpy as np
# # from pathlib import Path
# # from datetime import datetime, timedelta
# from delphi.core.data import TSData, TSLoader, TSPipeline


# # create a lot of dummy data with various time spans, frequencies, etc
# # create various dummy comfig.yaml files, load these files with UserConfig
# # test for assert Raise messages
# # test for correct date range after cropping
# # test for zero trimming (need data that has that)

# # write temp config.yaml
# # write temp data in csv and parquet
# # do test, then erase everything


# def test_dataloader_load_data():
#     # Generate dummy data
#     data1 = pd.DataFrame(
#         {"A": range(100), "B": range(100, 200)}, index=pd.date_range("2021-01-01", periods=100)
#     )
#     data2 = pd.DataFrame({"C": range(200, 300)}, index=pd.date_range("2021-01-01", periods=100))

#     # Generate temporary files
#     with tempfile.NamedTemporaryFile(
#         mode="w+", suffix=".csv"
#     ) as file1, tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as file2:
#         data1.to_csv(file1, index_label="Date")
#         data2.to_csv(file2, index_label="Date")

#         file1.flush()
#         file2.flush()

#         # Generate dummy config
#         config = {
#             "data_sources": [file1.name, file2.name],
#             "data_date_start": "2021-01-01",
#             "data_date_stop": "2021-04-10",
#             "target": {"name": "A"},
#             "trim_initial_zeros": False,
#             "trim_low_corr_variables": False,
#         }

#         dataloader = TSLoader(config)
#         dataloader.load_data()

#         assert isinstance(dataloader.raw_data, pd.DataFrame)
#         assert dataloader.raw_data.shape == (100, 3)
#         assert dataloader.target_column == "A"


# @pytest.fixture
# def config_schema():
#     return ConfigSchema(
#         project_name="data_test",
#         version_name="v_test",
#         project_dir="/project/dir",
#         data_sources=["data1", "data2"],
#         target={"var": "TC", "var2": "Panamax"},
#         model="TFTModel",
#         model_params={
#             "input_chunk_length": 6,
#             "output_chunk_length": 4,
#             "hidden_size": 16,
#             "lstm_layers": 1,
#         },
#         trainer_params={
#             "n_epochs": 500,
#             "batch_size": 64,
#             "learning_rate": 0.005,
#             "optimizer": "Adam",
#         },
#         num_samples=1,
#         n_jobs=8,
#         split=0.85,
#         data_date_start="2000",
#         data_date_stop="max",
#         covariates_preprocess={
#             "MinMaxScaler": {"feature_range": [0, 1]},
#             "PCA": {"n_components": 0.95},
#         },
#         target_preprocess={},
#     )


# @pytest.fixture
# def data_loader():
#     return TSLoader(config_schema)


# @pytest.fixture
# def ts_data(data_loader):
#     data_loader.load_data()
#     return TSData(data_loader.raw_data)


# @pytest.fixture
# def ts_pipeline():
#     return TSPipeline(config_schema)


# def test_load_data(data_loader):
#     data_loader.load_data()
#     assert isinstance(data_loader.raw_data, pd.DataFrame)
#     assert not data_loader.raw_data.empty


# def test_find_target_variable(data_loader):
#     data_loader.load_data()
#     target_column = data_loader.find_target_variable()
#     assert target_column == "target"


# def test_crop_date_range(data_loader):
#     data_loader.load_data()
#     cropped_data = data_loader.crop_date_range(data_loader.raw_data)
#     assert not cropped_data.empty
#     assert isinstance(cropped_data, pd.DataFrame)


# def test_trim_initial_zeros(data_loader):
#     data_loader.load_data()
#     data_loader.trim_initial_zeros()
#     assert not data_loader.raw_data.empty
#     assert isinstance(data_loader.raw_data, pd.DataFrame)


# def test_split_target_from_covariates(ts_data):
#     ts_data.split_target_from_covariates("target")
#     assert ts_data.target_name == "target"
#     assert not ts_data.X.empty
#     assert not ts_data.y.empty
#     assert isinstance(ts_data.X, pd.DataFrame)
#     assert isinstance(ts_data.y, pd.DataFrame)


# def test_split_past_future_covariates(ts_data):
#     ts_data.split_target_from_covariates("target")
#     ts_data.split_past_future_covariates()
#     assert ts_data.X_past is not None or ts_data.X_future is not None


# def test_split_train_test(ts_data):
#     ts_data.split_target_from_covariates("target")
#     ts_data.split_past_future_covariates()
#     ts_data.split_train_test(0.7)
#     assert not ts_data.y_train.empty
#     assert not ts_data.y_test.empty
#     assert not ts_data.y_val.empty


# def test_save_covariates(ts_data, tmp_path):
#     ts_data.split_target_from_covariates("target")
#     ts_data.save_covariates(tmp_path)
#     yaml_file = tmp_path.joinpath("used_covariates.yaml")
#     assert yaml_file.exists()


# def test_load_covariates(ts_data, tmp_path):
#     ts_data.split_target_from_covariates("target")
#     ts_data.save_covariates(tmp_path)
#     ts_data.load_covariates(tmp_path.joinpath("used_covariates.yaml"))
#     assert ts_data.cov_names is not None


# def test_to_timeseries(ts_data):
#     ts_data.split_target_from_covariates("target")
#     ts_data.split_past_future_covariates()
#     ts_data.split_train_test(0.7)
#     ts_data.to_timeseries()
#     assert not ts_data.y_train.empty
#     assert not ts_data.y_test.empty
#     assert not ts_data.y_val.empty


# def test_pipeline_build(ts_pipeline):
#     ts_pipeline.build()
#     assert ts_pipeline.past_cov_pipe is not None
#     assert ts_pipeline.future_cov_pipe is not None
#     assert ts_pipeline.target_pipe is not None


# def test_pipeline_fit(ts_data, ts_pipeline):
#     pass
