from collections import defaultdict

from scalars import Scalar, is_scalar
from states import BaseState, State
from toolbox import assert_list_or_tuple, simplify, replace_var


class BaseOperator:
    def __init__(self, left, right):
        """Represents a single term of an operator, i.e. |left><right|,
        where left and right are :class:`~.states.BaseState`'s.
        """
        if not all(isinstance(s, BaseState) for s in [left, right]):
            raise TypeError(f"Both left and right should be of type BaseState, not {type(left)} or {type(right)}")
        self._left = left
        self._right = right

    def __eq__(self, other):
        self._assert_class(other)
        return (self._left == other._left) and (self._right == other._right)

    def __hash__(self):
        return hash((hash(self._left), hash(self._right)))

    def __mul__(self, other):
        if not self._mul_compatible(other):
            raise TypeError(f"other ({other}) is not multiplication compatible with self ({self})")
        if not isinstance(other, State):
            return NotImplemented
        new_state = State([])
        for base_state, scalar in other._terms.items():
            new_state._terms[self._left] += self._right.inner_product(base_state) * scalar

        return new_state

    def __str__(self):
        return f"Op[{self._left}{self._right._bra_str()}]"

    def __repr__(self):
        return f"{self._class__.__name__}({repr(self._left)}, {repr(self._right)})"

    def _mul_compatible(self, other):
        """Used to check if an operator or state is compatible for multiplication.

        For example if they act on the same number of qubits.
        """
        if isinstance(other, self.__class__):
            return self._right._compatible(other._left)
        elif isinstance(other, State):
            return self._right._compatible(next(iter(other._terms)))
        elif isinstance(other, BaseState):
            return self._right._compatible(other)
        return False

    def _add_compatible(self, other):
        """Used to check if an operator is compatible for addition.

        For example if they act on the same number of qubits.
        """
        if not isinstance(other, self.__class__):
            return False
        return self._left._compatible(other._left) and self._right._compatible(other._right)

    def to_operator(self):
        """Converts the base operator to an operator with a single term."""
        return Operator([self._left, self._right])

    def _assert_class(self, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError(f"other is not of type {self.__class__}, but {type(other)}")


class Operator:
    def __init__(self, base_ops=None, scalars=None):
        # NOTE assuming that all scalars can add int()
        self._terms = defaultdict(int)
        if base_ops is None:
            return
        assert_list_or_tuple(base_ops)
        if scalars is None:
            scalars = [1] * len(base_ops)
        else:
            assert_list_or_tuple(scalars)
            if len(base_ops) != len(scalars):
                raise ValueError(f"number of base operators ({len(base_ops)})"
                                 f"and scalars ({len(scalars)}) are not equal")

        for op_term, scalar in zip(base_ops, scalars):
            if not isinstance(op_term, BaseOperator):
                raise TypeError(f"elements of base_ops should be instances of BaseOperator, not {type(op_term)}")
            if not is_scalar(scalar):
                raise TypeError(f"elements of scalars should be instances of Scalar, not {type(op_term)}")
            self._terms[op_term] += scalar

        base_ops = list(self._terms.keys())
        # Check that all base_ops are pairwise compatible
        for i, bo1 in enumerate(base_ops):
            for j in range(i + 1, len(base_ops)):
                bo2 = base_ops[j]
                if not bo1._add_compatible(bo2):
                    raise ValueError(f"Base operators {bo1} and {bo2} are not compatible terms")

    def __mul__(self, other):
        if not self._mul_compatible(other):
            raise ValueError(f"operator not multiplication compatible with {other}")
        if isinstance(other, Scalar):
            return self._mul_scalar(other)
        elif isinstance(other, State):
            return self._mul_state(other)
        elif isinstance(other, Operator):
            return self._mul_operator(other)
        return NotImplemented

    def _mul_state(self, state):
        # compute output state for each term of the operator
        new_state = State([])
        for base_op, scalar in self._terms.items():
            new_state += scalar * (base_op * state)

        return new_state

    def _mul_scalar(self, scalar):
        new_op = Operator([])
        for base_op, c in self._terms.items():
            new_op._terms[base_op] += scalar * c

        return new_op

    def _mul_operator(self, operator):
        raise NotImplementedError()

    def __add__(self, other):
        if not self._add_compatible(other):
            raise ValueError(f"operator not addition compatible with {other}")
        # Do add
        new_op = Operator([])
        for op in [self, other]:
            for base_op, scalar in op._terms.items():
                new_op._terms[base_op] += scalar

        # Check for zero terms
        new_op._prune_zero_terms()

        return new_op

    def __radd__(self, other):
        return self + other

    def __len__(self):
        return len(self._terms)

    def __str__(self):
        to_print = ""
        for base_op, scalar in self._terms.items():
            to_print += f"{scalar}*{base_op} + "
        return to_print[:-3]

    def __repr__(self):
        base_ops = list(self._terms.keys())
        scalars = [self._terms[base_op] for base_op in base_ops]
        return f"{self.__class__.__name__}({repr(base_ops)}, {repr(scalars)})"

    def dagger(self):
        raise NotImplementedError()

    def simplify(self):
        new_op = Operator()
        for base_op, scalar in self._terms.items():
            new_op._terms[base_op] = simplify(scalar)

        new_op._prune_zero_terms()

        return new_op

    def replace_var(self, old_variable, new_variable):
        new_op = Operator()
        for base_op, scalar in self._terms.items():
            new_base_op = replace_var(base_op, old_variable, new_variable)
            new_scalar = replace_var(scalar, old_variable, new_variable)
            new_op._terms[new_base_op] = new_scalar

        return new_op

    def _prune_zero_terms(self):
        to_remove = []
        for base_op, scalar in list(self._terms.items()):
            if scalar == 0:
                to_remove.append(base_op)

        for base_state in to_remove:
            self._terms.pop(base_state)

    def _mul_compatible(self, other):
        """Used to check if an operator or state is compatible for multiplication.

        For example if they act on the same number of qubits.
        """
        if isinstance(other, self.__class__) or isinstance(other, State):
            if len(self) == 0 or len(other) == 0:
                return True

            # NOTE we only need to check one of the terms
            self_term = next(iter(self._terms.keys()))
            other_term = next(iter(other._terms.keys()))
            return self_term._mul_compatible(other_term)
        return False

    def _add_compatible(self, other):
        """Used to check if an operator is compatible for addition.

        For example if they act on the same number of qubits.
        """
        if not isinstance(other, self.__class__):
            return False
        if len(self) == 0 or len(other) == 0:
            return True

        # NOTE we only need to check one of the terms
        self_term = next(iter(self._terms.keys()))
        other_term = next(iter(other._terms.keys()))
        return self_term._add_compatible(other_term)


def outer_product(left, right):
    """Creates an opertor based on the outer product of left and right, i.e. |left><right|."""
    scalars = []
    base_ops = []
    for l_base_state, l_scalar in left._terms.items():
        for r_base_state, r_scalar in right._terms.items():
            scalars.append(l_scalar * r_scalar.conjugate())
            base_ops.append(BaseOperator(l_base_state, r_base_state))

    return Operator(base_ops, scalars)
