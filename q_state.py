from scalars import ComplexScalar
from states import BaseState


class BaseQuditState(BaseState):
    def __init__(self, digits):
        if not isinstance(digits, str):
            raise TypeError(f"digits should be a string, not {type(digits)}")
        self._assert_digits(digits)
        self._digits = digits

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._digits == other._digits

    def __hash__(self):
        return hash(self._digits)

    def __str__(self):
        return f"|{self._digits}>"

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._digits)})"

    def _compatible(self, other):
        if not isinstance(other, self.__class__):
            return False
        return len(self._digits) == len(other._digits)

    def inner_product(self, other):
        self._assert_class(other)
        if not self._compatible(other):
            raise ValueError("Can only do inner product between states on the same number of qubits")
        if self == other:
            return ComplexScalar(1)
        else:
            return ComplexScalar(0)

    def tensor_product(self, other):
        self._assert_class(other)
        return self.__class__(self._digits + other._digits)

    def _bra_str(self):
        return f"<{self._digits}|"

    def _assert_class(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"other is not of type {self.__class__}, but {type(other)}")

    def get_variables(self):
        return set([])

    def _assert_digits(self, digits):
        if not set(digits) <= {f'{i}' for i in range(10)}:
            raise ValueError(f"digits should contain only '0' to '9', not {set(digits)}")


class BaseQubitState(BaseQuditState):
    def _assert_digits(self, digits):
        if not set(digits) <= {'0', '1'}:
            raise ValueError(f"digits should contain only '0' and '1', not {set(digits)}")
