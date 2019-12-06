import numpy as np

from q_state import BaseQubitState
from operators import outer_product


def test_mul_state():
    z0 = BaseQubitState("0").to_state()
    z1 = BaseQubitState("1").to_state()
    x0 = (z0 + z1) * (1 / np.sqrt(2))
    x1 = (z0 - z1) * (1 / np.sqrt(2))
    H = outer_product(x0, z0) + outer_product(x1, z1)

    # Check the inner product after applying H
    assert np.isclose((H*z0).inner_product(z0), 1 / np.sqrt(2))
    assert np.isclose((H*z0).inner_product(z1), 1 / np.sqrt(2))
    assert np.isclose((H*z0).inner_product(x0), 1)
    assert np.isclose((H*z0).inner_product(x1), 0)

    assert np.isclose((H*x0).inner_product(z0), 1)
    assert np.isclose((H*x0).inner_product(z1), 0)
    assert np.isclose((H*x0).inner_product(x0), 1 / np.sqrt(2))
    assert np.isclose((H*x0).inner_product(x1), 1 / np.sqrt(2))
