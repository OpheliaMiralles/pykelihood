import pytest
from scipy.stats import expon, genextreme
import pandas as pd


@pytest.fixture(scope="session")
def datasets():
    return (
        pd.Series(genextreme.rvs(size=1000, c=-0.2)),
        pd.Series(expon.rvs(size=1000)),
    )
