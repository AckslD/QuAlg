import abc
from collections import defaultdict

from scalars import ComplexScalar, is_scalar
from toolbox import assert_list_or_tuple


class BaseState(abc.ABC):
    @abc.abstractmethod
    def __eq__(self, other):
        pass

    @abc.abstractmethod
    def __hash__(self):
        pass

    @abc.abstractmethod
    def inner_product(self, other):
        pass

    @abc.abstractmethod
    def _compatible(self, other):
        """Used to check if two states are compatible.

        For example if they have the same number of qubits.
        """
        pass


class BaseQubitState(BaseState):
    def __init__(self, binary):
        if not isinstance(binary, str):
            raise TypeError(f"binary should be a string, not {type(binary)}")
        if not set(binary) <= {'0', '1'}:
            raise ValueError(f"binary should contain only '0' and '1', not {set(binary)}")
        self._binary = binary

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._binary == other._binary

    def __hash__(self):
        return hash(self._binary)

    def __str__(self):
        return f"|{self._binary}>"

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._binary)})"

    def _compatible(self, other):
        if not isinstance(other, self.__class__):
            return False
        return len(self._binary) == len(other._binary)

    def inner_product(self, other):
        self._assert_class(other)
        if not self._compatible(other):
            raise ValueError("Can only do inner product between states on the same number of qubits")
        if self == other:
            return ComplexScalar(1)
        else:
            return ComplexScalar(0)

    def _bra_str(self):
        return f"<{self._binary}|"

    def to_state(self):
        """Converts the base state to a state with a single term."""
        return State([self])

    def _assert_class(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"other is not of type {self.__class__}, but {type(other)}")


class State:
    def __init__(self, base_states, scalars=None):
        assert_list_or_tuple(base_states)
        if scalars is None:
            scalars = [1] * len(base_states)
        else:
            assert_list_or_tuple(scalars)
            if len(base_states) != len(scalars):
                raise ValueError("number of base_states and scalars must be equal")

        # NOTE assuming that all scalars can add int()
        self._terms = defaultdict(int)
        for base_state, scalar in zip(base_states, scalars):
            if not isinstance(base_state, BaseState):
                raise TypeError(f"base_states needs to be of class BaseState, not {type(base_state)}")
            if not is_scalar(scalar):
                raise TypeError(f"scalars needs to be of class Scalar, not {type(scalar)}")
            self._terms[base_state] += scalar

        # Check that all states are pairwise compatible
        base_states = list(self._terms.keys())
        for i, bs1 in enumerate(base_states):
            for j in range(i + 1, len(base_states)):
                bs2 = base_states[j]
                if not bs1._compatible(bs2):
                    raise ValueError(f"States {bs1} and {bs2} are not compatible terms")

    def __eq__(self, other):
        raise NotImplementedError()

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError(f"addition is not implemented for {type(other)}")
        # Check that the states are compatible
        # NOTE we only need to check the first terms of the two states.
        if not self._compatible(other):
            raise ValueError(f"other ({other}) is not compatible with self ({self})")
        new_state = State([])
        for state in [self, other]:
            for base_state, scalar in state._terms.items():
                new_state._terms[base_state] += scalar

        # Check if there are any zero terms
        new_state._prune_zero_states()

        return new_state

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return other + (-1 * self)

    def __mul__(self, other):
        if not is_scalar(other):
            return NotImplemented
        new_state = State([])
        for base_state, scalar in self._terms.items():
            new_state._terms[base_state] = scalar * other

        return new_state

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        to_return = ""
        for base_state, scalar in self._terms.items():
            to_return += f"{scalar}*{base_state} + "
        return to_return[:-3]

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(list(self._terms.items()))})"

    def __len__(self):
        return len(self._terms)

    def inner_product(self, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError(f"inner product is not implemented for {type(other)}")
        if not self._compatible(other):
            raise ValueError(f"other ({other}) is not compatible with self ({self})")
        # NOTE if BaseState are assumed to be orthogonal be don't need to do the
        # product of base states.
        inner = 0
        for self_base_state, self_scalar in self._terms.items():
            for other_base_state, other_scalar in other._terms.items():
                inner += (self_scalar.conjugate() * other_scalar) * self_base_state.inner_product(other_base_state)

        return inner

    def _compatible(self, other):
        if not isinstance(other, State):
            return False
        if len(self) == 0 or len(other) == 0:
            return True

        # NOTE we only need to check one of the terms
        self_term = next(iter(self._terms.keys()))
        other_term = next(iter(other._terms.keys()))
        return self_term._compatible(other_term)

    def _prune_zero_states(self):
        to_remove = []
        for base_state, scalar in list(self._terms.items()):
            if scalar == 0:
                to_remove.append(base_state)

        for base_state in to_remove:
            self._terms.pop(base_state)

    def _bra_str(self):
        to_return = ""
        for base_state, scalar in self._terms.items():
            to_return += f"{scalar.conjugate()}*{base_state._bra_str()} + "
        return to_return[:-3]
