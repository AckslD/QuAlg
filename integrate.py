from copy import copy
from collections import namedtuple

from toolbox import assert_str, replace_var, simplify, get_variables
from scalars import is_number, DeltaFunction, SumOfScalars, ProductOfScalars, InnerProductFunction,\
    SingleVarFunctionScalar, Scalar, assert_is_scalar


IntegrateResult = namedtuple("IntegrateResult", ["scalar", "applied"])


class _Integration(Scalar):

    def __init__(self, scalar, variable):
        assert_is_scalar(scalar)
        assert_str(variable)
        if not isinstance(scalar, ProductOfScalars):
            raise TypeError(f"scalar should be ProductOfScalars, not {type(scalar)}")
        if not all(factor.has_variable(variable) for factor in scalar):
            raise ValueError("all factors should have the integration term")
        self._scalar = scalar
        self._variable = variable

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        # TODO check with swapping the integration variable since this shouldn't matter
        return self._scalar == other._scalar and self._variable == other._variable

    def __hash__(self):
        return hash(self._key())

    def __str__(self):
        return f"S_{self._variable}{{{self._scalar}}}"

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._scalar)}, {repr(self._variable)}"

    def __copy__(self):
        return self.__class__(self._scalar, self._variable)

    def conjugate(self):
        return self.__class__(self._scalar.conjugate(), self._variable)

    def simplify(self):
        new_scalar = copy(self)
        for evaluation in EVALUATIONS:
            new_scalar = evaluation(new_scalar)
            if not isinstance(new_scalar, _Integration):
                return new_scalar
        return new_scalar

    def has_variable(self, variable):
        # raise NotImplementedError()
        if variable == self._variable:
            return False
        return self._scalar.has_variable(variable)

    def _key(self):
        return (self._scalar, self._variable)


def integrate(scalar, variable=None):
    # TODO needed?
    scalar = simplify(scalar)
    if variable is None:
        return integrate(scalar, get_variables(scalar))
    if isinstance(variable, set):
        new_scalar = scalar
        for v in variable:
            new_scalar = integrate(new_scalar, v)
        return new_scalar
    assert_str(variable)
    if isinstance(scalar, SumOfScalars):
        new_scalar = sum(integrate(s, variable) for s in scalar._terms)
    elif isinstance(scalar, ProductOfScalars):
        # Split factors based on if they contain the integration variable or not
        var_factors = []
        other_factors = []
        for factor in scalar._factors:
            if isinstance(factor, Scalar) and factor.has_variable(variable):
                var_factors.append(factor)
            else:
                other_factors.append(factor)
        new_scalar = ProductOfScalars(other_factors + [_Integration(ProductOfScalars(var_factors), variable)])
    elif is_number(scalar):
        new_scalar = scalar
    else:
        raise NotImplementedError(f"integrate not implemented for type {type(scalar)}")

    return simplify(new_scalar)


def _evaluate_delta_function(integration_scalar):
    integrand = integration_scalar._scalar
    variable = integration_scalar._variable
    # Find the delta functions containing this variable
    deltas = [(i, d) for (i, d) in enumerate(integrand._factors)
              if isinstance(d, DeltaFunction) and variable in d._vars]
    if len(deltas) == 0:
        return integration_scalar
    elif len(deltas) > 2:
        raise RuntimeError("Two delta functions with the same variable")
    # Replace variables with the variable to the other variable in the delta function
    i, delta = deltas[0]
    # Get the other variable in the delta function
    other_var = next(v for v in delta._vars if v != variable)
    integrand = replace_var(integrand, old_variable=variable, new_variable=other_var)
    # Remove the delta funtion (recall that replace_var creates a copy)
    integrand._factors.pop(i)

    return integrand


def _find_norm_identities(integration_scalar):
    """Finds integrals which are the norm of a function, i.e. 1"""
    integrand = integration_scalar._scalar
    # TODO generalise this
    if len(integrand) == 2:
        if all(isinstance(s, SingleVarFunctionScalar) for s in integrand):
            if integrand[0] == integrand[1].conjugate():
                return 1
    return integration_scalar


def _find_function_inner_products(integration_scalar):
    """Finds integrals which evaluate to the inner product of functions."""
    integrand = integration_scalar._scalar
    # TODO generalise this
    if len(integrand) == 2:
        if all(isinstance(s, SingleVarFunctionScalar) for s in integrand):
            return InnerProductFunction(integrand[0]._func_name, integrand[1]._func_name)
    return integration_scalar


# These evaluations are used when integrating
EVALUATIONS = [
    _evaluate_delta_function,
    _find_norm_identities,
    _find_function_inner_products,
]


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


if __name__ == '__main__':
    test_integrate()
    # test_integrate2()
