import pytest
import numpy as np

from qualg.measure import measure
from qualg.q_state import BaseQubitState
from qualg.operators import outer_product


def test_measurement():
    s0 = BaseQubitState("0").to_state()
    s1 = BaseQubitState("1").to_state()
    h0 = (s0 + s1) * (1 / np.sqrt(2))

    P0 = outer_product(s0, s0)
    P1 = outer_product(s1, s1)
    kraus_ops = {0: P0, 1: P1}

    # Measure |0>
    meas_res = measure(s0, kraus_ops)
    assert meas_res.outcome == 0
    assert np.isclose(meas_res.probability, 1)

    # Measure |1>
    meas_res = measure(s1, kraus_ops)
    assert meas_res.outcome == 1
    assert np.isclose(meas_res.probability, 1)

    # Measure |+>
    meas_res = measure(h0, kraus_ops)
    assert np.isclose(meas_res.probability, 1 / 2)


@pytest.mark.parametrize("state, kraus_ops, error", [
    (None, {}, TypeError),
    (BaseQubitState("0").to_state(), None, TypeError),
    (BaseQubitState("0").to_state(), {"0": 0, "1": 1}, TypeError),
])
def test_faulty(state, kraus_ops, error):
    with pytest.raises(error):
        measure(state, kraus_ops)


def test_non_complete_kraus():
    s0 = BaseQubitState("0").to_state()
    s1 = BaseQubitState("1").to_state()
    op = outer_product(s0, s0)
    with pytest.raises(ValueError):
        measure(s1, {"0": op})
