from qualg.toolbox import simplify
from qualg.scalars import SingleVarFunctionScalar, DeltaFunction, SumOfScalars


def test_product_of_scalars():
    f1 = SingleVarFunctionScalar("f", "x")
    f2 = SingleVarFunctionScalar("f", "y")
    d = DeltaFunction("x", "y")
    expr = f1 * f2.conjugate() * d
    print(expr)
    assert set(expr) == set([f1, f2.conjugate(), d])


def test_expand():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')
    c = SingleVarFunctionScalar('c', 'x')
    d = SingleVarFunctionScalar('d', 'x')

    expr = (a + b) * (c + d)
    expr = simplify(expr.expand())
    print(expr)
    assert isinstance(expr, SumOfScalars)
    assert len(expr) == 4
    print(expr[0])
    assert len(expr[0]) == 2


def test_simplify():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')
    c = SingleVarFunctionScalar('c', 'x')

    expr = (a + b) * (c + 0)
    expr = expr.simplify()
    print(expr)
    assert isinstance(expr, SumOfScalars)
    assert len(expr) == 2
    print(expr[0])
    assert len(expr[0]) == 2
