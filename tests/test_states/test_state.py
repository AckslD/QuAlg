import pytest
import numpy as np

from states import State
from q_state import BaseQubitState


@pytest.mark.parametrize("input, num_terms, error", [
    (BaseQubitState("0"), None, TypeError),
    ((1, BaseQubitState("0")), None, TypeError),
    ([(None, BaseQubitState("0"))], None, TypeError),
    ([(1, None)], None, TypeError),
    ([BaseQubitState("0")], 1, None),
    ([
        BaseQubitState("0"),
        BaseQubitState("1"),
    ], 2, None),
    ([
        BaseQubitState("0"),
        BaseQubitState("0"),
    ], 1, None),
    ([
        BaseQubitState("0"),
        BaseQubitState("00"),
    ], None, ValueError),
])
def test_init(input, num_terms, error):
    if error is not None:
        with pytest.raises(error):
            State(input)
    else:
        s = State(input)
        assert len(s._terms) == num_terms


# def test_eq(self, other):
#     pass

# def test_add(self, other):
#     pass

def test_mul():
    s = BaseQubitState("00").to_state() + BaseQubitState("11").to_state()
    x = 1 / np.sqrt(2)
    s2 = s * x
    s3 = x * s
    assert str(s2) == f"{x}*|00> + {x}*|11>"
    assert str(s3) == f"{x}*|00> + {x}*|11>"

# def test_str(self, other):
#     pass

# def test_repr(self, other):
#     pass

# def inner_product(self, other):
#     pass
