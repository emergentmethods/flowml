import numpy as np
import pandas as pd
import pytest

@pytest.fixture(scope="function")
def mocked_default_values(mocker):
    mock_default_values = mocker.patch("flowdapt.compute.artifacts._get_values_from_context")
    mock_default_values.return_value = ("default", "memory", "", {})

    return mock_default_values


# make a dummy array
def dummy_array(rows: int, cols: int, withnans: bool = True) -> np.ndarray:

    arr = np.random.rand(rows, cols)
    # fake nans
    if withnans:
        arr = np.where(arr < 0.01, np.nan, arr)

    return arr


def dummy_pandas_df(rows: int, cols: int, withnans: bool = True) -> pd.DataFrame:

    df = pd.DataFrame(np.random.rand(rows, cols)) * 35
    # fake features
    df.columns = [f"%-{col}" for col in df.columns]
    # fake label
    df = df.set_axis([*df.columns[:-1], '&-a'], axis=1)
    # fake nans
    if withnans:
        df = df.mask(df < 0.01)

    return df