from typing import Any, Mapping, Optional, Sequence, Union

from darts.dataprocessing.transformers import FittableDataTransformer, InvertibleDataTransformer
from darts.logging import get_logger  # , raise_if
from darts.timeseries import TimeSeries
import numpy as np

# import pandas as pd
from sklearn.decomposition import PCA as skPCA


logger = get_logger(__name__)


# TODO test this class


class PCA(FittableDataTransformer, InvertibleDataTransformer):
    def __init__(
        self,
        n_components: Optional[int] = None,
        global_fit: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """
        Principal Component Analysis (PCA) data transformer.

        Applies PCA from scikit-learn to the TimeSeries data. The transformation is applied
        independently for each dimension (component) of the time series. For stochastic series,
        it is done jointly over all samples, effectively merging all samples of a component
        in order to compute the transform.

        Notes
        -----
        The transformer will not act on the series' static covariates. This has to be done either
        before constructing the series or later on by extracting the covariates, transforming the
        values, and then reapplying them to the series. For this, see TimeSeries properties
        `TimeSeries.static_covariates` and method `TimeSeries.with_static_covariates()`

        Parameters
        ----------
        n_components
            The number of components to keep after PCA. If None, all components are kept.
        global_fit
            Optionally, whether all of the `TimeSeries` passed to the `fit()` method should be
            used to fit a *single* set of parameters, or if a different set of parameters should
            be independently fitted to each provided `TimeSeries`. If `True`, then a
            `Sequence[TimeSeries]` is passed to `ts_fit` and a single set of parameters is fitted
            using all of the provided `TimeSeries`. If `False`, then each `TimeSeries` is
            individually passed to `ts_fit`, and a different set of fitted parameters if yielded
            for each of these fitting operations. See `FittableDataTransformer` documentation for
            further details.
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a
            ``Sequence[TimeSeries]`` is passed as input, parallelising operations regarding
            different ``TimeSeries``. Defaults to `1` (sequential).
            Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up
            increasing the total required amount of time.
        verbose
            Whether to print operations progress

        """
        super().__init__(
            name="PCA",
            n_jobs=n_jobs,
            verbose=verbose,
            mask_components=True,
            global_fit=global_fit,
        )
        self.n_components = n_components

    @staticmethod
    def ts_fit(
        series: Union[TimeSeries, Sequence[TimeSeries]], params: Mapping[str, Any], *args, **kwargs
    ) -> skPCA:
        n_components = params["fixed"]["n_components"]

        if isinstance(series, TimeSeries):
            series = [series]

        # Concatenate all samples
        vals = np.concatenate([ts.values(copy=True) for ts in series], axis=0)
        pca = skPCA(n_components=n_components)
        pca.fit(vals)

        return pca

    @staticmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any], **kwargs) -> TimeSeries:
        pca = params["fitted"]

        # Apply PCA transformation
        transformed_vals = pca.transform(series.values(copy=True))

        # Create a new TimeSeries with the transformed values
        return TimeSeries.from_times_and_values(
            times=series.time_index, values=transformed_vals, columns=series.columns
        )

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries, params: Mapping[str, Any], **kwargs
    ) -> TimeSeries:
        pca = params["fitted"]

        # Apply PCA inverse transformation
        inv_transformed_vals = pca.inverse_transform(series.values(copy=True))

        # Create a new TimeSeries with the inverse transformed values
        return TimeSeries.from_times_and_values(
            times=series.time_index, values=inv_transformed_vals, columns=series.columns
        )
