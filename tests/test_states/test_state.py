import pytest
import numpy as np

from qualg.states import State
from qualg.q_state import BaseQubitState
from qualg.toolbox import simplify
from qualg.scalars import ProductOfScalars


@pytest.mark.parametrize("input, scalars, num_terms, error", [
    (BaseQubitState("0"), None, None, TypeError),
    ((1, BaseQubitState("0")), None, None, TypeError),
    ([(None, BaseQubitState("0"))], None, None, TypeError),
    ([(1, None)], None, None, TypeError),
    ([BaseQubitState("0")], None, 1, None),
    ([
        BaseQubitState("0"),
        BaseQubitState("1"),
    ], None, 2, None),
    ([
        BaseQubitState("0"),
        BaseQubitState("0"),
    ], None, 1, None),
    ([
        BaseQubitState("0"),
        BaseQubitState("00"),
    ], None, None, ValueError),
    ([
        BaseQubitState("0"),
        BaseQubitState("1"),
    ], 1, None, TypeError),
    ([
        BaseQubitState("0"),
        BaseQubitState("1"),
    ], [1], None, ValueError),
    ([
        BaseQubitState("0"),
        BaseQubitState("1"),
    ], [1, None], None, TypeError),
])
def test_init(input, scalars, num_terms, error):
    if error is not None:
        with pytest.raises(error):
            State(input, scalars=scalars)
    else:
        s = State(input, scalars=scalars)
        assert len(s._terms) == num_terms


def test_mul():
    s = BaseQubitState("00").to_state() + BaseQubitState("11").to_state()
    x = 1 / np.sqrt(2)
    s2 = s * x
    s3 = x * s
    assert str(s2) == f"{x}*|00> + {x}*|11>"
    assert str(s3) == f"{x}*|00> + {x}*|11>"


def test_tensor_product():
    s = (BaseQubitState('0').to_state() + BaseQubitState('1').to_state()) * (1 / np.sqrt(2))
    s_tensor = s @ s
    expected = sum((BaseQubitState(f"{i}{j}").to_state() for i in range(2) for j in range(2)), State())
    expected *= 1 / 2

    expected == s_tensor


def test_simplify():
    prod = ProductOfScalars()
    prod._factors = [2, 3, 5]

    bs = BaseQubitState('0')
    s = State([bs], scalars=[prod])
    assert s.get_scalar(bs) != 30
    s = simplify(s)
    assert s.get_scalar(bs) == 30
