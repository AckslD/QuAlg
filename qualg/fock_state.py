"""
Contains classes for representing in excited states of some bosonic mode.
The creation/annihilation operators have a symbolic variable which can for example
be the frequency of the excited state.
"""
from collections import defaultdict
from itertools import permutations

from qualg.scalars import DeltaFunction
from qualg.states import BaseState
from qualg.toolbox import assert_str, assert_list_or_tuple, replace_var


class FockOp:
    def __init__(self, mode, variable, creation=True):
        """
        Represents an creation/annihilation operator in a given mode with a given variable
        representing for example the frequency of the excitation.

        Parameters
        ----------

        mode: str
            The mode of the excitation
        variable: str
            The variable describing the continous variable
        creation : bool
            Whether this is a creation operator or not (annihilation).
        """
        assert_str(mode)
        assert_str(variable)
        self._mode = mode
        self._variable = variable
        self._creation = creation

    def _key(self):
        return self._mode, self._variable, self._creation

    def __eq__(self, other):
        if not isinstance(other, FockOp):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __str__(self):
        dag = "+" if self._creation else ""
        return f"{self._mode}{dag}({self._variable})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self._mode}, {self._variable}, {self._creation})"

    def dagger(self):
        """
        Complex conjugate of the operator.
        """
        return FockOp(self._mode, self._variable, not self._creation)

    def replace_var(self, old_variable, new_variable):
        """
        Replaces a variable with another.
        """
        var = self._variable
        if old_variable == var:
            var = new_variable
        return self.__class__(self._mode, var, creation=self._creation)

    def get_variables(self):
        """
        Returns the variable of this operator.
        """
        return set([self._variable])


class FockOpProduct:
    def __init__(self, fock_ops=None):
        """
        A product of excitation operators (:class:`~.FockOp`).

        Parameters
        ----------
        fock_ops : list of :class:`~.FockOp`
            The product of fock operators.
        """
        self._fock_ops = defaultdict(int)
        if fock_ops is None:
            return
        assert_list_or_tuple(fock_ops)
        for fock_op in fock_ops:
            assert_fock_op(fock_op)
            # TODO we only allow these to be creation ops for now
            if not fock_op._creation:
                raise NotImplementedError
            self._fock_ops[fock_op] += 1

    def __mul__(self, other):
        new_op = FockOpProduct()
        if isinstance(other, FockOp):
            for fock_op, count in self._fock_ops.items():
                new_op._fock_ops[fock_op] += count
            new_op._fock_ops[other] += 1
            return new_op
        elif isinstance(other, FockOpProduct):
            for fock_op, count in self._fock_ops.items():
                new_op._fock_ops[fock_op] += count
            for fock_op, count in other._fock_ops.items():
                new_op._fock_ops[fock_op] += count

        return new_op

    def __str__(self):
        to_print = ""
        for fock_op, count in self._fock_ops.items():
            to_print += f"{fock_op}^{count} * "
        return to_print[:-3]

    def _key(self):
        return tuple(sorted(self._fock_ops.items(), key=lambda x: x[0]._key()))

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._key() == other._key()

    def dagger(self):
        """
        Complex conjugate of the operator.
        """
        new_op = FockOpProduct()
        for fock_op, count in self._fock_ops.items():
            new_op._fock_ops[fock_op.dagger()] = count

        return new_op

    def variables_in_mode(self, mode):
        """
        Get the variables in a given mode.
        """
        variables = []
        for fock_op, count in self._fock_ops.items():
            if fock_op._mode == mode:
                for _ in range(count):
                    variables.append(fock_op._variable)

        return variables

    def variables_by_modes(self):
        """
        Get a dictionay of variables per modes.
        """
        variables = defaultdict(list)
        for fock_op, count in self._fock_ops.items():
            mode = fock_op._mode
            for _ in range(count):
                variables[mode].append(fock_op._variable)

        return variables

    def replace_var(self, old_variable, new_variable):
        """
        Replaces a variable with another.
        """
        new_fock_op_product = FockOpProduct()
        for fock_op, count in self._fock_ops.items():
            new_fock_op_product._fock_ops[replace_var(fock_op, old_variable, new_variable)] = count
        return new_fock_op_product

    def get_variables(self):
        """
        Returns the variable of this operator.
        """
        vars = set([])
        for fock_op in self._fock_ops:
            vars |= fock_op.get_variables()

        return vars


class BaseFockState(BaseState):
    def __init__(self, fock_ops=None):
        """
        A base state represented by excitations from vacuum.

        Parameters
        ----------
        fock_ops : :class:`.FockOpProduct` or list of :class:`~.FockOp`
            The product of fock operators.
        """
        if isinstance(fock_ops, FockOpProduct):
            self._fock_op_product = fock_ops
        else:
            self._fock_op_product = FockOpProduct(fock_ops)

    def _key(self):
        return self._fock_op_product._key()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __str__(self):
        to_print = ""
        for fock_op, count in self._fock_op_product._fock_ops.items():
            to_print += f"{fock_op}^{count}"
        return to_print + "|0>"

    @property
    def shape(self):
        return None

    def inner_product(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError()
        l_vars_by_mode = self._fock_op_product.variables_by_modes()
        r_vars_by_mode = other._fock_op_product.variables_by_modes()
        all_modes = set(l_vars_by_mode.keys()) | set(r_vars_by_mode.keys())
        scalar = 1
        for mode in all_modes:
            l_vars = l_vars_by_mode.get(mode, [])
            r_vars = r_vars_by_mode.get(mode, [])
            if len(l_vars) != len(r_vars):
                return 0
            factor = 0
            for perm_r_vars in permutations(r_vars):
                term = 1
                for l_var, r_var in zip(l_vars, perm_r_vars):
                    term *= DeltaFunction(l_var, r_var)
                factor += term
            scalar *= factor

        return scalar

    def tensor_product(self, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError(f"fock tensor product is not implemented for {type(other)}")
        prod = self._fock_op_product * other._fock_op_product
        return BaseFockState(fock_ops=prod)

    def replace_var(self, old_variable, new_variable):
        """
        Replaces a variable with another.
        """
        return self.__class__(replace_var(self._fock_op_product, old_variable, new_variable))

    def get_variables(self):
        """
        Returns the variable of this operator.
        """
        return self._fock_op_product.get_variables()

    def _compatible(self, other):
        if not isinstance(other, BaseFockState):
            return False
        return True

    def _bra_str(self):
        to_print = ""
        for fock_op, count in self._fock_op_product._fock_ops.items():
            to_print += f"{fock_op.dagger()}^{count}"
        return "<0|" + to_print


def assert_fock_op(op):
    """
    Asserts that an object is a fock operator.
    """
    if not isinstance(op, FockOp):
        raise TypeError(f"operator should be of type FockOp, not {type(op)}")
