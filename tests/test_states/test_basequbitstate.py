import pytest

from q_state import BaseQubitState


@pytest.mark.parametrize("input, error", [
    ("1010", None),
    (10, TypeError),
    (None, TypeError),
    (1.0, TypeError),
    ("012", ValueError),
    ("O1O", ValueError),  # letter O instead of digit 0
])
def test_init(input, error):
    if error is not None:
        with pytest.raises(error):
            BaseQubitState(input)
    else:
        s = BaseQubitState(input)
        assert s._digits == input


@pytest.mark.parametrize("state1, state2, expected, error", [
    (BaseQubitState("0"), BaseQubitState("0"), True, None),
    (BaseQubitState("1"), BaseQubitState("0"), False, None),
    (BaseQubitState("0000"), BaseQubitState("0"), False, None),
    (BaseQubitState("0000"), BaseQubitState("0000"), True, None),
    (BaseQubitState("0001"), BaseQubitState("0000"), False, None),
    (BaseQubitState("0000"), "0000", False, None),
])
def test_eq(state1, state2, expected, error):
    if error is not None:
        print(state1 == state2)
        with pytest.raises(error):
            state1 == state2
    else:
        assert (state1 == state2) == expected


@pytest.mark.parametrize("state1, state2, expected, error", [
    (BaseQubitState("0"), BaseQubitState("0"), 1, None),
    (BaseQubitState("1"), BaseQubitState("0"), 0, None),
    (BaseQubitState("0000"), BaseQubitState("0000"), 1, None),
    (BaseQubitState("0001"), BaseQubitState("0000"), 0, None),
    (BaseQubitState("0000"), BaseQubitState("000"), None, ValueError),
    (BaseQubitState("0000"), "0000", None, TypeError),
])
def test_inner_product(state1, state2, expected, error):
    if error is not None:
        with pytest.raises(error):
            state1.inner_product(state2)
    else:
        inner = state1.inner_product(state2)
        assert inner == expected


@pytest.mark.parametrize("state1, state2, expected", [
    (BaseQubitState("0"), BaseQubitState("0"), True),
    (BaseQubitState("1"), BaseQubitState("0"), True),
    (BaseQubitState("0000"), BaseQubitState("0000"), True),
    (BaseQubitState("000"), BaseQubitState("0000"), False),
    (BaseQubitState("000"), "000", False),
])
def test_compatible(state1, state2, expected):
    assert state1._compatible(state2) == expected
