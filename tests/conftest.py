import numpy as np
import pandas as pd
import pytest
from scipy.stats import expon, genextreme


@pytest.fixture(scope="session")
def datasets():
    return (
        pd.Series(genextreme.rvs(size=1000, c=-0.2)),
        pd.Series(expon.rvs(size=1000)),
    )


@pytest.fixture(scope="session")
def dataset():
    return pd.Series(np.random.randn(1000))


@pytest.fixture(scope="session")
def matrix_data():
    return pd.DataFrame(np.random.randn(2, 3), columns=("first", "second", "third"))
