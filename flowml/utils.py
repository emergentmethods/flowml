import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from typing import Tuple
from pandas import DataFrame

from flowdapt.lib.logger import get_logger

logger = get_logger(__name__)


def make_train_test_datasets(
        features, labels, weight_factor, data_params
) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike,
           npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """
    Wrapper for SKLearn's train_test_split()
    """
    weights = set_weights_higher_recent(len(features), weight_factor)

    (
        train_features,
        test_features,
        train_labels,
        test_labels,
        train_weights,
        test_weights,
    ) = train_test_split(
        features,
        labels,
        weights,
        **data_params
    )

    return train_features, test_features, train_labels, test_labels, train_weights, test_weights


def set_weights_higher_recent(num_weights: int, wfactor: float) -> npt.ArrayLike:
    """
    Set weights so that recent data is more heavily weighted during
    training than older data.
    """
    if wfactor == 0:
        weights = np.ones(num_weights)
    else:
        weights = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]
    return weights


def reduce_dataframe_footprint(df: DataFrame) -> DataFrame:
    """
    Ensure all values are float32 in the incoming dataframe.
    :param df: Dataframe to be converted to float/int 32s
    :return: Dataframe converted to float/int 32s
    """

    logger.debug(f"Memory usage of dataframe is "
                 f"{df.memory_usage().sum() / 1024**2:.2f} MB")

    df_dtypes = df.dtypes
    for column, dtype in df_dtypes.items():
        if dtype == np.float64:
            df_dtypes[column] = np.float32
        elif dtype == np.int64:
            df_dtypes[column] = np.int32
    df = df.astype(df_dtypes)

    logger.debug(f"Memory usage after optimization is: "
                 f"{df.memory_usage().sum() / 1024**2:.2f} MB")

    return df
