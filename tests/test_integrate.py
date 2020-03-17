import pytest

from qualg.toolbox import has_variable
from qualg.integrate import integrate
from qualg.scalars import SingleVarFunctionScalar, DeltaFunction, InnerProductFunction,\
    ProductOfScalars, SumOfScalars


def test_integrate():
    f1 = SingleVarFunctionScalar("f", "x")
    f2 = SingleVarFunctionScalar("f", "y")
    d = DeltaFunction("x", "y")
    expr = f1 * f2.conjugate() * d
    res = integrate(expr, "x")
    print(res)
    assert set(res) == set([
        SingleVarFunctionScalar("f", "y", False),
        SingleVarFunctionScalar("f", "y", True),
    ])


def test_integrate_equal_terms():
    f1 = SingleVarFunctionScalar("f", "x")
    f2 = SingleVarFunctionScalar("f", "y")
    f3 = f1 * f2.conjugate()
    d = DeltaFunction("x", "y")
    expr = (f3 + f3) * d
    res = integrate(expr, "x")
    print(res)
    assert isinstance(res, ProductOfScalars)
    assert set(res) == set([
        2,
        SingleVarFunctionScalar("f", "y", False),
        SingleVarFunctionScalar("f", "y", True),
    ])


def test_integrate_sum():
    f1 = SingleVarFunctionScalar("f", "x")
    f2 = SingleVarFunctionScalar("g", "x")
    d = DeltaFunction("x", "y")
    expr = (f1 + f2) * d
    res = integrate(expr, "x")
    print(res)
    assert isinstance(res, SumOfScalars)
    assert set(res) == set([
        SingleVarFunctionScalar("f", "y"),
        SingleVarFunctionScalar("g", "y"),
    ])


def test_integrate_single_function():
    f = SingleVarFunctionScalar("f", "x")
    d = DeltaFunction("x", "y")
    expr = f * d
    res = integrate(expr, "x")
    print(res)
    assert res == SingleVarFunctionScalar("f", "y")


def test_integrate_all_vars():
    f = SingleVarFunctionScalar("f", "x")
    g = SingleVarFunctionScalar("g", "y").conjugate()
    d = DeltaFunction("x", "y")
    expr = f * g * d
    res = integrate(expr)
    print(res)
    assert res == InnerProductFunction("f", "g")


def test_integrate_same_function():
    fx = SingleVarFunctionScalar("f", "x")
    fy = SingleVarFunctionScalar("f", "y").conjugate()
    d = DeltaFunction("x", "y")
    expr = fx * fy * d
    res = integrate(expr)
    print(res)
    assert res == 1


@pytest.mark.parametrize("var", [
    1,
    InnerProductFunction('f', 'g'),
])
def test_integrate_scalar_no_var(var):
    assert var == integrate(var, "x")


def test_has_variable():
    fx = SingleVarFunctionScalar("f", "x")
    fy = SingleVarFunctionScalar("f", "y").conjugate()
    expr = fx * fy
    res = integrate(expr, "x")
    print(res)
    assert not has_variable(res, "x")
    assert has_variable(res, "y")
