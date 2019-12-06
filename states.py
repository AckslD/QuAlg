import abc
from collections import defaultdict

from scalars import is_scalar
from toolbox import assert_list_or_tuple, simplify, replace_var, get_variables, is_zero


class BaseState(abc.ABC):
    @abc.abstractmethod
    def __eq__(self, other):
        pass

    @abc.abstractmethod
    def __hash__(self):
        pass

    def __matmul__(self, other):
        return self.tensor_product(other)

    @abc.abstractmethod
    def inner_product(self, other):
        pass

    @abc.abstractmethod
    def tensor_product(self, other):
        pass

    @abc.abstractmethod
    def _compatible(self, other):
        """Used to check if two states are compatible.

        For example if they have the same number of qubits.
        """
        pass

    def to_state(self):
        """Converts the base state to a state with a single term."""
        return State([self])

    @abc.abstractmethod
    def _bra_str(self):
        pass


class State:
    def __init__(self, base_states=None, scalars=None):
        if base_states is None:
            self._terms = defaultdict(int)
            return
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
        # Add this terms
        for base_state, scalar in self._terms.items():
            new_state._terms[base_state] = scalar
        # Add the other terms
        for base_state, scalar in other._terms.items():
            new_state._terms[base_state] += scalar

        # Check if there are any zero terms
        new_state._prune_zero_terms()

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

    def __matmul__(self, other):
        return self.tensor_product(other)

    def __iter__(self):
        return iter(self._terms.items())

    def get_scalar(self, base_state):
        """Returns the scalar of the given base_state"""
        return self._terms.get(base_state, 0)

    def inner_product(self, other, first_replace_var=True):
        if not isinstance(other, self.__class__):
            raise NotImplementedError(f"inner product is not implemented for {type(other)}")
        if not self._compatible(other):
            raise ValueError(f"other ({other}) is not compatible with self ({self})")
        # NOTE if BaseState are assumed to be orthogonal be don't need to do the
        # product of base states.
        if first_replace_var:
            other = replace_var(other)
        inner = 0
        for self_base_state, self_scalar in self._terms.items():
            for other_base_state, other_scalar in other._terms.items():
                factors = [
                    self_scalar.conjugate(),
                    other_scalar,
                    self_base_state.inner_product(other_base_state),
                ]
                if any(is_zero(factor) for factor in factors):
                    continue
                inner += (self_scalar.conjugate() * other_scalar) * self_base_state.inner_product(other_base_state)

        return inner

    def tensor_product(self, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError(f"tensor product is not implemented for {type(other)}")
        tensor = State()
        for self_base_state, self_scalar in self._terms.items():
            for other_base_state, other_scalar in other._terms.items():
                tensor += State(base_states=[self_base_state.tensor_product(other_base_state)],
                                scalars=[self_scalar * other_scalar])

        return tensor

    def simplify(self):
        new_state = State()
        for base_state, scalar in self._terms.items():
            new_state._terms[base_state] = simplify(scalar)

        new_state._prune_zero_terms()

        return new_state

    def replace_var(self, old_variable, new_variable):
        new_state = State()
        for base_state, scalar in self._terms.items():
            new_base_state = replace_var(base_state, old_variable, new_variable)
            new_scalar = replace_var(scalar, old_variable, new_variable)
            new_state._terms[new_base_state] = new_scalar

        return new_state

    def get_variables(self):
        vars = set([])
        for term in self:
            for part in term:
                vars |= get_variables(part)

        return vars

    def _compatible(self, other):
        if not isinstance(other, State):
            return False
        if len(self) == 0 or len(other) == 0:
            return True

        # NOTE we only need to check one of the terms
        self_term = next(iter(self._terms.keys()))
        other_term = next(iter(other._terms.keys()))
        return self_term._compatible(other_term)

    def _prune_zero_terms(self):
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
