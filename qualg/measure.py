"""Contains function :func:`~.measure` for measuring states"""
import random
import numpy as np
from collections import namedtuple

from qualg.scalars import is_number
from qualg.states import State
from qualg.operators import Operator


MeasurementResult = namedtuple("MeasurementResult", ["outcome", "probability", "post_meas_state"])


def measure(state, kraus_ops):
    """Measures a state with a given list of Kraus operators describing a POVM.

    Parameters
    ----------
    state : :class:`~.states.State`
        The state to be measured.
    kraus_ops : dict
        Dictionary containing the Kraus operators describing the POVM as values and
        the outcomes as keys.

    Returns
    -------
    :class:`~.MeasurementResult`
        The namedtuple returned contains:
        * `outcome`: The outcome of the measurement, i.e. the key of the Kraus operator.
        * `probability`: The probability of the outcome.
        * `post_meas_state`: The post-measurement state.

    Warning
    -------
    There is no check that the given Kraus operators are actually a valid POVM.
    """
    if not isinstance(state, State):
        raise TypeError("state should be a State")
    if not isinstance(kraus_ops, dict):
        raise TypeError("kraus_ops should be a dict")
    r = random.random()
    offset = 0
    for outcome, kraus_op in kraus_ops.items():
        if not isinstance(kraus_op, Operator):
            raise TypeError("the values of kraus_ops should be Operator")
        post_state = kraus_op * state
        p = post_state.inner_product(post_state)
        if not is_number(p):
            raise NotImplementedError("Cannot perform measurement when inner product are not numbers")
        if p < 0:
            raise ValueError("Seems the Kraus operators does not form positive operators")
        if offset <= r <= offset + p:
            post_state = post_state * (1 / np.sqrt(p))
            return MeasurementResult(outcome, p, post_state)
        offset += p
    raise ValueError("Seems the Kraus operators does not sum up to one")
