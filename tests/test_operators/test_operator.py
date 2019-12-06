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


def test_to_numpy_matrix():
    phi = (1 / np.sqrt(2)) * (BaseQubitState("00").to_state() + BaseQubitState("11").to_state())
    op = outer_product(phi, phi)
    expected = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2
    print(op)
    print(op.to_numpy_matrix())
    assert np.all(np.isclose(op.to_numpy_matrix(), expected))
