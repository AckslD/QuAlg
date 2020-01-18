"""
Contains classes for qubit and qubit base states.
"""
from qualg.states import BaseState


class BaseQuditState(BaseState):
    def __init__(self, digits, base=2):
        """
        A qudit base state.

        Parameters
        ----------
        digits : str
            A string of the digits representing the base state, e.g "010".
            Which digits in the range 0..(base-1).
        base (optional) : int
            How many basis states there are per position, e.g. 2 (defualt) for qubits.
        """
        if base >= 10:
            # TODO
            raise NotImplementedError("'base' must be lower than 10")
        if not isinstance(digits, str):
            raise TypeError(f"digits should be a string, not {type(digits)}")
        self._base = base
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

    def __len__(self):
        return len(self._digits)

    @property
    def shape(self):
        return (self._base ** len(self),)

    def _compatible(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (len(self._digits) == len(other._digits)) and (self._base == other._base)

    def inner_product(self, other):
        self._assert_class(other)
        if not self._compatible(other):
            raise ValueError("Can only do inner product between states on the same number of qubits")
        if self == other:
            return 1
        else:
            return 0

    def tensor_product(self, other):
        self._assert_class(other)
        return self.__class__(self._digits + other._digits)

    def _vector_index(self):
        """Specifies the index in an actual vector."""
        return int(self._digits, base=self._base)

    def _bra_str(self):
        return f"<{self._digits}|"

    def _assert_class(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"other is not of type {self.__class__}, but {type(other)}")

    def get_variables(self):
        return set([])

    def _assert_digits(self, digits):
        if not set(digits) <= {f'{i}' for i in range(self._base)}:
            raise ValueError(f"digits should contain only '0' to '{self._base - 1}', not {set(digits)}")


class BaseQubitState(BaseQuditState):
    def __init__(self, digits):
        """
        A qubit base state. (same as :class:`~.BaseQuditState` except that 'base' is fixed to 2)

        Parameters
        ----------
        digits : str
            A string of the digits representing the base state, e.g "010".
            Which digits in the range 0..1.
        """
        super().__init__(digits, base=2)
