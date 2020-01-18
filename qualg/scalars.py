"""
Containing classes for various scalars.
Used for example as output when taking inner-product of states.
In particular when the output is not a number but a symbolic variable of function.
"""

import abc
import math
from copy import copy
from collections import defaultdict
from itertools import product

from qualg.toolbox import assert_list_or_tuple, assert_str, expand, simplify, is_zero, replace_var, is_one,\
    get_variables, has_variable


def is_number(n):
    """Check if something is a number (int, float or complex)"""
    return any(isinstance(n, tp) for tp in [int, float, complex])


def is_scalar(n):
    """Check if something is considered a scalar (number of :class:`~.Scalar`)"""
    return is_number(n) or isinstance(n, Scalar)


class Scalar(abc.ABC):
    """
    Base-class for all scalars.

    Meant to be subclassed.
    """
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __add__(self, other):
        # Check if one is zero
        if is_zero(other):
            return copy(self)
        if is_zero(self):
            return copy(other)

        if not is_scalar(other):
            return NotImplemented
        if isinstance(other, SumOfScalars):
            return other + self
        # Check if multiples
        self_multiple, self_scalar = _get_multiple_of_scalar(self)
        other_multiple, other_scalar = _get_multiple_of_scalar(other)
        if self_scalar == other_scalar:
            return copy(self_scalar) * (self_multiple + other_multiple)

        return SumOfScalars([self, other])

    def __mul__(self, other):
        if not is_scalar(other):
            return NotImplemented
        if isinstance(other, ProductOfScalars):
            return other * self
        return ProductOfScalars([self, other])

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def has_variable(self, variable):
        """Checks if scalar depends on a given variable."""
        return False

    def get_variables(self):
        """
        Returns the variable of this operator.
        """
        return set([])

    @abc.abstractmethod
    def conjugate(self):
        """Complex conjugate"""
        pass

    @abc.abstractmethod
    def _key(self):
        pass


class SingleVarFunctionScalar(Scalar):
    def __init__(self, func_name, variable, conjugate=False):
        """Represents a function with a single symbolic variable, e.g. f(x).

        Parameters
        ----------
        func_name : str
            Name of function.
        variable : str
            Name of variable.
        conjugate (optional) : bool
            If the functions is conjugated (default: `False`)
        """
        assert_str(func_name)
        assert_str(variable)
        self._func_name = func_name
        self._variable = variable
        self._conjugate = conjugate

    def __str__(self):
        conj = "*" if self._conjugate else ""
        return f"{self._func_name}{conj}({self._variable})"

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._func_name)}, {repr(self._variable)}, {self._conjugate})"

    def conjugate(self):
        return SingleVarFunctionScalar(self._func_name, self._variable, not self._conjugate)

    def replace_var(self, old_variable, new_variable):
        var = self._variable
        if old_variable == var:
            var = new_variable
        return self.__class__(self._func_name, var, conjugate=self._conjugate)

    def get_variables(self):
        return set([self._variable])

    def is_zero(self):
        return False

    def is_one(self):
        return False

    def has_variable(self, variable):
        return variable == self._variable

    def _key(self):
        return (self._func_name, self._variable, self._conjugate)


class InnerProductFunction(Scalar):
    def __init__(self, func_name1, func_name2):
        """Represents the inner product of two functions.

        Parameters
        ----------
        func_name1 : str
            Name of first function
        func_name2 : str
            Name of second function
        """
        assert_str(func_name1)
        assert_str(func_name2)
        self._func_names = sorted([func_name1, func_name2])

    def __str__(self):
        return f"<{self._func_names[0]}|{self._func_names[1]}>"

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._func_names[0])}, {repr(self._func_names[1])})"

    def conjugate(self):
        return self.__class__(self._func_names[0], self._func_names[1])

    def is_zero(self):
        return False

    def is_one(self):
        return False

    def _key(self):
        return tuple(self._func_names)


class DeltaFunction(Scalar):
    def __init__(self, var1, var2):
        """Delta function between two variables, e.g. d(x - y)

        Parameters
        ----------
        var1 : str
            First variable
        var2 : str
            First variable
        """
        assert_str(var1)
        assert_str(var2)
        self._assert_different(var1, var2)
        self._vars = [var1, var2]

    def conjugate(self):
        return DeltaFunction(*self._vars)

    def __str__(self):
        v1, v2 = self._vars
        return f"D[{v1}-{v2}]"

    def __repr__(self):
        v1, v2 = self._vars
        return f"{self.__class__.__name__}({v1}, {v2})"

    def replace_var(self, old_variable, new_variable):
        new_vars = list(self._vars)
        if old_variable in new_vars:
            new_vars.remove(old_variable)
            new_vars.append(new_variable)
        self._assert_different(*new_vars)
        return self.__class__(*new_vars)

    def get_variables(self):
        return set(self._vars)

    def is_zero(self):
        return False

    def is_one(self):
        return False

    def has_variable(self, variable):
        return variable in self._vars

    def _key(self):
        return frozenset(self._vars)

    @staticmethod
    def _assert_different(var1, var2):
        if var1 == var2:
            raise ValueError(f"Variables in a delta function needs to be different, not {var1} and {var2}")


class ProductOfScalars(Scalar):
    def __init__(self, scalars=None):
        """Product of some number of scalars.

        Parameters
        ----------
        scalars : None or list of :class:`~.scalar`
            The factors of the product.
        """
        self._factors = [1]
        if scalars is None:
            return
        assert_list_or_tuple(scalars)
        for scalar in scalars:
            assert_is_scalar(scalar)
            if is_number(scalar):
                self._factors[0] *= scalar
            else:
                self._factors.append(scalar)

    def __str__(self):
        to_print = ""
        for scalar in self:
            to_print += f"{scalar}*"
        return to_print[:-1]

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(list(iter(self)))})"

    def __mul__(self, other):
        if not is_scalar(other):
            return NotImplemented
        if is_number(other):
            return ProductOfScalars([self._factors[0] * other] + self._factors[1:])
        if isinstance(other, ProductOfScalars):
            new_scalar = copy(self)
            for factor in other:
                new_scalar *= factor
            return new_scalar
        # Check if the multiplication between other and any factor can be simplified
        for i, factor in enumerate(self._factors):
            if i == 0:
                continue
            prod = factor * other
            if not isinstance(prod, ProductOfScalars):
                return ProductOfScalars([prod] + self._factors[:i] + self._factors[i + 1:])
        return ProductOfScalars(self._factors + [other])

    def _start_index(self):
        return 1 if (self._factors[0] == 1 and len(self._factors) > 1) else 0

    def __len__(self):
        start = self._start_index()
        return len(self._factors[start:])

    def __iter__(self):
        start = self._start_index()
        return iter(self._factors[start:])

    def __getitem__(self, i):
        start = self._start_index()
        return self._factors[start + i]

    def __setitem__(self, i, value):
        start = self._start_index()
        self._factors[start + i] = value

    def conjugate(self):
        return ProductOfScalars([scalar.conjugate() for scalar in self])

    def atoms(self):
        atoms = []
        for s in self:
            if _is_sequenced_scalar(s):
                atoms += s.atoms()
            else:
                atoms.append(s)
        return atoms

    def replace_var(self, old_variable, new_variable):
        new_factors = []
        for s in self:
            new_factors.append(replace_var(s, old_variable, new_variable))
        return self.__class__(new_factors)

    def get_variables(self):
        vars = set([])
        for factor in self:
            vars |= get_variables(factor)

        return vars

    def expand(self):
        if not any(isinstance(s, SumOfScalars) for s in self._factors):
            # No factor is a sum
            return ProductOfScalars(list(iter(self)))
        expandable_factors = []
        for s in self:
            if isinstance(s, SumOfScalars):
                expandable_factors.append(s._terms)
            else:
                expandable_factors.append([s])

        # Expand the product
        return SumOfScalars([ProductOfScalars(term) for term in product(*expandable_factors)])

    def is_zero(self):
        for f in self._factors:
            if isinstance(f, float):
                if math.isclose(f, 0, abs_tol=1e-16):
                    return True
            else:
                if is_zero(f):
                    return True
        return False

    def is_one(self):
        return all(is_one(s) for s in self._factors)

    def simplify(self, full=True):
        new_scalar = 1
        for factor in self:
            if is_zero(factor):
                return 0
            if not is_one(factor):
                new_scalar *= simplify(factor)
        new_scalar = expand(new_scalar)
        if isinstance(new_scalar, SumOfScalars):
            return simplify(new_scalar)

        if _is_sequenced_scalar(new_scalar):
            if len(new_scalar) == 0:
                return 0
            if len(new_scalar) == 1:
                return simplify(new_scalar[0])

        if isinstance(new_scalar, ProductOfScalars):
            if any(isinstance(s, ProductOfScalars) for s in new_scalar):
                raise RuntimeError("factors of product should not be products")
            if full:
                changed = True
                while changed:
                    changed = new_scalar._combine_factors()
        return new_scalar

    def _combine_factors(self):
        """Test if factors can be combined"""
        for i, factor1 in enumerate(self._factors):
            if i == 0:
                continue
            if is_number(factor1):
                self._factors[0] += factor1
                self._factors.pop(i)
                return True
            for j in range(i + 1, len(self._factors)):
                factor2 = self._factors[j]
                prod = factor1 * factor2
                if not isinstance(prod, ProductOfScalars) or len(prod._factors) <= 2:
                    self._factors[i] = prod
                    self._factors.pop(j)
                    return True
        return False

    def has_variable(self, variable):
        return any(has_variable(factor, variable) for factor in self)

    def _key(self):
        factors_with_multi = defaultdict(int)
        for factor in self:
            factors_with_multi[factor] += 1
        return frozenset(factors_with_multi.items())


class SumOfScalars(Scalar):
    def __init__(self, scalars=None):
        """Sum of some number of scalars.

        Parameters
        ----------
        scalars : None or list of :class:`~.scalar`
            The terms of the sum.
        """
        self._terms = [0]
        if scalars is None:
            return
        assert_list_or_tuple(scalars)
        for scalar in scalars:
            assert_is_scalar(scalar)
            if is_number(scalar):
                self._terms[0] += scalar
            else:
                self._terms.append(scalar)

    def __str__(self):
        to_print = ""
        for scalar in self:
            to_print += f"{scalar} + "
        return "(" + to_print[:-3] + ")"

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(list(iter(self._terms)))})"

    def __add__(self, other):
        # Check if one is zero
        if is_zero(other):
            return copy(self)
        if is_zero(self):
            return copy(other)

        if not is_scalar(other):
            return NotImplemented
        if is_number(other):
            return SumOfScalars([self._terms[0] + other] + self._terms[1:])
        if isinstance(other, SumOfScalars):
            new_scalar = copy(self)
            for term in other:
                new_scalar += term
            return new_scalar
        # Check if the addition between other and any term can be simplified
        for i, term in enumerate(self._terms):
            if i == 0:
                continue
            sm = term + other
            if not isinstance(sm, SumOfScalars):
                return SumOfScalars([sm] + self._terms[:i] + self._terms[i + 1:])
        return SumOfScalars(self._terms + [other])

    def _start_index(self):
        return 1 if self._terms[0] == 0 else 0

    def __len__(self):
        start = self._start_index()
        return len(self._terms[start:])

    def __iter__(self):
        start = self._start_index()
        return iter(self._terms[start:])

    def __getitem__(self, i):
        start = self._start_index()
        return self._terms[start + i]

    def __setitem__(self, i, value):
        start = self._start_index()
        self._terms[start + i] = value

    def conjugate(self):
        return SumOfScalars([scalar.conjugate() for scalar in self._terms])

    def atoms(self):
        atoms = []
        for s in self:
            if _is_sequenced_scalar(s):
                atoms += s.atoms()
            else:
                atoms.append(s)
        return atoms

    def replace_var(self, old_variable, new_variable):
        new_terms = []
        for s in self:
            new_terms.append(replace_var(s, old_variable, new_variable))
        return self.__class__(new_terms)

    def get_variables(self):
        vars = set([])
        for term in self:
            vars |= get_variables(term)

        return vars

    def is_zero(self):
        return all(is_zero(s) for s in self)

    def expand(self):
        return sum((expand(term) for term in self))

    def simplify(self, full=True):
        new_scalar = 0
        expanded = self.expand()

        if not isinstance(expanded, SumOfScalars):
            return simplify(expanded)
        if is_number(expanded):
            return expanded
        for term in expanded:
            if not is_zero(term):
                new_scalar += simplify(term)

        if _is_sequenced_scalar(new_scalar):
            if len(new_scalar) == 0:
                new_scalar = 0
            if len(new_scalar) == 1:
                new_scalar = simplify(new_scalar[0])

        if isinstance(new_scalar, SumOfScalars):
            if any(isinstance(s, SumOfScalars) for s in new_scalar):
                raise RuntimeError("terms of sum should not be sum after expand")
            if full:
                changed = True
                while changed:
                    changed = new_scalar._combine_terms()
        return new_scalar

    def _combine_terms(self):
        """Test if terms can be combined"""
        for i, term1 in enumerate(self._terms):
            if i == 0:
                continue
            if is_number(term1):
                self._terms[0] += term1
                self._terms.pop(i)
                return True
            for j in range(i + 1, len(self._terms)):
                term2 = self._terms[j]
                sm = term1 + term2
                if not isinstance(sm, SumOfScalars) or len(sm._terms) <= 2:
                    self._terms[i] = sm
                    self._terms.pop(j)
                    return True
        return False

    def has_variable(self, variable):
        return any(has_variable(factor, variable) for factor in self)

    def _key(self):
        terms_with_multi = defaultdict(int)
        for term in self:
            terms_with_multi[term] += 1
        return frozenset(terms_with_multi.items())


def assert_is_scalar(n):
    """Asserts that something is a scalar"""
    if not is_scalar(n):
        raise TypeError(f"variable ({n}) should be a scalar, not {type(n)}")


def _get_multiple_of_scalar(scalar):
    if isinstance(scalar, ProductOfScalars):
        if len(scalar._factors) == 2:
            return tuple(scalar._factors)
        else:
            return scalar._factors[0], ProductOfScalars(scalar._factors[1:])
    return 1, scalar


def _is_sequenced_scalar(scalar):
    return any(isinstance(scalar, tp) for tp in [ProductOfScalars, SumOfScalars])
