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
