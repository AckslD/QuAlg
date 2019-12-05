import pytest

from scalars import ComplexScalar


@pytest.mark.parametrize("input, error", [
    (0, None),
    (0., None),
    (1j, None),
    (None, TypeError),
    ("1", TypeError),
])
def test_init(input, error):
    if error is not None:
        with pytest.raises(error):
            ComplexScalar(input)
    else:
        s = ComplexScalar(input)
        assert s._c == input


@pytest.mark.parametrize("c1, c2, expected, error", [
    (ComplexScalar(0), ComplexScalar(0), True, None),
    (ComplexScalar(1), ComplexScalar(0), False, None),
    (ComplexScalar(0), 0, True, None),
    (ComplexScalar(1), 0, False, None),
    (ComplexScalar(1), None, False, None),
    (ComplexScalar(1), "1", False, None),
])
def test_eq(c1, c2, expected, error):
    if error is not None:
        with pytest.raises(error):
            c1 == c2
    else:
        assert (c1 == c2) == expected


@pytest.mark.parametrize("c1, c2, expected, error", [
    (ComplexScalar(0), ComplexScalar(0), ComplexScalar(0), None),
    (ComplexScalar(1), ComplexScalar(2), ComplexScalar(2), None),
    (ComplexScalar(2), 3, ComplexScalar(6), None),
    (ComplexScalar(1), 0, ComplexScalar(0), None),
    (ComplexScalar(1), None, None, TypeError),
    (ComplexScalar(1), "1", None, TypeError),
])
def test_mul(c1, c2, expected, error):
    if error is not None:
        with pytest.raises(error):
            c1 * c2
    else:
        assert (c1 * c2) == expected


@pytest.mark.parametrize("c1, c2, expected, error", [
    (ComplexScalar(0), ComplexScalar(0), ComplexScalar(0), None),
    (ComplexScalar(1), ComplexScalar(2), ComplexScalar(3), None),
    (ComplexScalar(2), 3, ComplexScalar(5), None),
    (ComplexScalar(1), 0, ComplexScalar(1), None),
    (ComplexScalar(1), None, None, TypeError),
    (ComplexScalar(1), "1", None, TypeError),
])
def test_add(c1, c2, expected, error):
    if error is not None:
        with pytest.raises(error):
            c1 + c2
    else:
        assert (c1 + c2) == expected
