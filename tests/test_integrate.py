from qualg.integrate import integrate
from qualg.scalars import SingleVarFunctionScalar, DeltaFunction


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


def test_integrate2():
    f1 = SingleVarFunctionScalar("f", "x")
    f2 = SingleVarFunctionScalar("f", "y")
    f3 = f1 * f2.conjugate()
    d = DeltaFunction("x", "y")
    expr = (f3 + f3) * d
    res = integrate(expr, "x")
    print(res)
    assert set(res) == set([
        2,
        SingleVarFunctionScalar("f", "y", False),
        SingleVarFunctionScalar("f", "y", True),
    ])
