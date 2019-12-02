import abc
import operator
from itertools import product

from toolbox import assert_list_or_tuple, assert_str, expand, simplify, is_zero


class Scalar(abc.ABC):
    @abc.abstractmethod
    def __eq__(self, other):
        pass

    def __add__(self, other):
        # TODO can be smarter if self or other is SumOfScalars
        if not is_scalar(other):
            return NotImplemented
        return SumOfScalars([self, other])

    def __mul__(self, other):
        # TODO can be smarter if self or other is ProductOfScalars
        if not is_scalar(other):
            return NotImplemented
        return ProductOfScalars([self, other])

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    @abc.abstractmethod
    def conjugate(self):
        pass


class ComplexScalar(Scalar):
    def __init__(self, c=0):
        self.assert_number(c)

        self._c = c

    def __eq__(self, other):
        return self._do_binary_op(other, operator.eq, bool)

    def __mul__(self, other):
        return self._do_binary_op(other, operator.mul, self.__class__)

    def __add__(self, other):
        return self._do_binary_op(other, operator.add, self.__class__)

    def __str__(self):
        return str(self._c)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._c)})"

    def _do_binary_op(self, other, op, output_class):
        if is_number(other):
            return output_class(op(self._c, other))
        elif isinstance(other, self.__class__):
            return output_class(op(self._c, other._c))
        else:
            return NotImplemented

    def conjugate(self):
        if isinstance(self._c, complex):
            return ComplexScalar(self._c.conjugate())
        else:
            return ComplexScalar(self._c)

    @staticmethod
    def assert_number(c):
        if not is_number(c):
            raise TypeError(f"c should be of type int, float or complex, not {type(c)}")


class SingleVarFunctionScalar(Scalar):
    def __init__(self, func_name, variable, conjugate=False):
        assert_str(func_name)
        assert_str(variable)
        self._func_name = func_name
        self._variable = variable
        self._conjugate = conjugate

    def __eq__(self, other):
        if not isinstance(other, SingleVarFunctionScalar):
            return NotImplemented
        return (self._func_name == other._func_name
                and self._variable == self._variable
                and self._conjugate == self._conjugate)

    def __str__(self):
        conj = "*" if self._conjugate else ""
        return f"{self._func_name}{conj}({self._variable})"

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._func_name)}, {repr(self._variable)}, {self._conjugate})"

    def conjugate(self):
        return SingleVarFunctionScalar(self._func_name, self._variable, not self._conjugate)

    def replace_var(self, old_variable, new_variable):
        if self._variable == old_variable:
            self._variable = new_variable

    def is_zero(self):
        return False


class DeltaFunction(Scalar):
    def __init__(self, var1, var2):
        assert_str(var1)
        assert_str(var2)
        self._vars = [var1, var2]

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return set(self._vars) == set(other._vars)

    def conjugate(self):
        return DeltaFunction(*self._vars)

    def __str__(self):
        v1, v2 = self._vars
        return f"D[{v1}-{v2}]"

    def __repr__(self):
        v1, v2 = self._vars
        return f"{self.__class__.__name__}({v1}, {v2})"

    def replace_var(self, old_variable, new_variable):
        if old_variable in self._vars:
            self._vars.remove(old_variable)
            self._vars.append(new_variable)

    def is_zero(self):
        return False


class ProductOfScalars(Scalar):
    def __init__(self, scalars=None):
        self._factors = []
        if scalars is None:
            return
        assert_list_or_tuple(scalars)
        self._factors = []
        for scalar in scalars:
            assert_is_scalar(scalar)
            self._factors.append(scalar)

    def __eq__(self, other):
        raise NotImplementedError()

    def __str__(self):
        to_print = ""
        for scalar in self._factors:
            factor = f"{scalar}"
            # Add brackets if scalar is a SumOfScalars
            if isinstance(scalar, SumOfScalars):
                factor = f"({factor})"
            to_print += f"{factor}*"
        return to_print[:-1]

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._factors)})"

    def __mul__(self, other):
        if not is_scalar(other):
            return NotImplemented
        if isinstance(other, ProductOfScalars):
            other_scalars = other._factors
        else:
            other_scalars = [other]
        return ProductOfScalars(self._factors + other_scalars)

    def conjugate(self):
        return ProductOfScalars([scalar.conjugate() for scalar in self._factors])

    def atoms(self):
        atoms = []
        for s in self._factors:
            if any(isinstance(s, tp) for tp in [ProductOfScalars, SumOfScalars]):
                atoms += s.atoms()
            else:
                atoms.append(s)
        return atoms

    def replace_var(self, old_variable, new_variable):
        for s in self._factors:
            if is_scalar_with_variable(s):
                s.replace_var(old_variable, new_variable)

    def expand(self):
        expandable_factors = []
        for s in self._factors:
            if isinstance(s, SumOfScalars):
                expandable_factors.append(s._terms)
            else:
                expandable_factors.append([s])

        # Expand the product
        return SumOfScalars([ProductOfScalars(term) for term in product(*expandable_factors)])

    def is_zero(self):
        return any(is_zero(s) for s in self._factors)

    def simplify(self):
        new_scalar = self.expand()
        return simplify(new_scalar)


class SumOfScalars(Scalar):
    def __init__(self, scalars=None):
        self._terms = []
        if scalars is None:
            return
        assert_list_or_tuple(scalars)
        for scalar in scalars:
            if not is_scalar(scalar):
                raise TypeError("entries of scalars should be of type int, float, complex or Scalar, "
                                f"not {type(scalar)}")
            self._terms.append(scalar)

    def __eq__(self, other):
        return NotImplemented

    def __str__(self):
        to_print = ""
        for scalar in self._terms:
            to_print += f"{scalar} + "
        return to_print[:-3]

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._terms)})"

    def __add__(self, other):
        if not is_scalar(other):
            return NotImplemented
        if isinstance(other, SumOfScalars):
            other_scalars = other._terms
        else:
            other_scalars = [other]
        return SumOfScalars(self._terms + other_scalars)

    def conjugate(self):
        return SumOfScalars([scalar.conjugate() for scalar in self._terms])

    def atoms(self):
        atoms = []
        for s in self._terms:
            if any(isinstance(s, tp) for tp in [ProductOfScalars, SumOfScalars]):
                atoms += s.atoms()
            else:
                atoms.append(s)
        return atoms

    def replace_var(self, old_variable, new_variable):
        for s in self._terms:
            if is_scalar_with_variable(s):
                s.replace_var(old_variable, new_variable)

    def is_zero(self):
        return all(s for s in self._factors)

    def expand(self):
        return sum((expand(term) for term in self._terms), SumOfScalars())

    def simplify(self):
        new_scalar = self.expand()
        return SumOfScalars([term for term in new_scalar._terms if not is_zero(term)])


def is_scalar_with_variable(scalar):
    types = [
        SingleVarFunctionScalar,
        DeltaFunction,
        SumOfScalars,
        ProductOfScalars,
    ]
    return any(isinstance(scalar, tp) for tp in types)


def integrate(scalar, variable):
    assert_str(variable)
    if isinstance(scalar, SumOfScalars):
        return sum(integrate(s) for s in scalar._terms)
    elif isinstance(scalar, ProductOfScalars):
        # Check if has delta
        for i, s in enumerate(scalar._factors):
            # TODO add checks for consistency
            if isinstance(s, DeltaFunction):
                if variable in s._vars:
                    other_var = list(s._vars - set([variable]))[0]
                    new_scalars = scalar._factors[:i] + scalar._factors[i+1:]
                    new_scalar = ProductOfScalars(new_scalars)
                    for atom in new_scalar.atoms():
                        atom.replace_var(old_variable=variable, new_variable=other_var)
                    return new_scalar


def is_number(n):
    return any(isinstance(n, tp) for tp in [int, float, complex])


def is_scalar(n):
    return is_number(n) or isinstance(n, Scalar)


def assert_is_scalar(n):
    if not is_scalar(n):
        raise TypeError(f"variable ({n}) should be a scalar, not {type(n)}")


def test_sum_of_scalars():
    f1 = SingleVarFunctionScalar("f", "x")
    f2 = SingleVarFunctionScalar("f", "y")
    d = DeltaFunction("x", "y")
    expr = f1 * f2.conjugate() * d
    print(expr)


def test_integrate():
    f1 = SingleVarFunctionScalar("f", "x")
    f2 = SingleVarFunctionScalar("f", "y")
    d = DeltaFunction("x", "y")
    expr = f1 * f2.conjugate() * d
    print(integrate(expr, "x"))


def test_integrate2():
    f1 = SingleVarFunctionScalar("f", "x")
    f2 = SingleVarFunctionScalar("f", "y")
    f3 = f1 * f2.conjugate()
    d = DeltaFunction("x", "y")
    expr = (f3 + f3) * d
    print(integrate(expr, "x"))


def test_expand():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')
    c = SingleVarFunctionScalar('c', 'x')
    d = SingleVarFunctionScalar('d', 'x')

    expr = (a + b) * (c + d)
    print(expr.expand())


def test_simplify():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')
    c = SingleVarFunctionScalar('c', 'x')
    zero = ComplexScalar(0)

    expr = (a + b) * (c + zero)
    print(expr.simplify())


if __name__ == '__main__':
    # test_sum_of_scalars()
    # test_integrate()
    # test_integrate2()
    # test_expand()
    test_simplify()
