import random

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
    return pd.DataFrame(np.random.randn(100, 3), columns=("first", "second", "third"))


@pytest.fixture(scope="session")
def categorical_data():
    return pd.Series(random.choices(["item1", "item2", "item3"], k=100))


@pytest.fixture(scope="session")
def categorical_data_boolean():
    return pd.Series(random.choices([True, False], k=100))
