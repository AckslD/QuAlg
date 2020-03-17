import pytest

from qualg.toolbox import simplify, replace_var, get_variables, has_variable, is_zero
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


def test_combine_factors():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')
    c = SingleVarFunctionScalar('c', 'x')

    prod = a * b * c
    prod[0] = 2
    prod[2] = 5
    prod = simplify(prod)
    assert prod == 10 * b


def test_replace_var_delta():
    d1 = DeltaFunction('x', 'y')
    d1 = replace_var(d1, 'y', 'z')
    d2 = DeltaFunction('x', 'z')

    assert d1 == d2

    with pytest.raises(ValueError):
        replace_var(d1, 'z', 'x')


def test_conjugate_delta():
    d1 = DeltaFunction('x', 'y')
    d2 = d1.conjugate()
    assert d1 == d2


def test_atoms_product():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')

    prod = 2 * a * b

    print(prod)
    atoms = prod.atoms()
    assert set([2, a, b]) == set(atoms)


def test_atoms_sum():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')

    sm = 2 + a + b

    print(sm)
    atoms = sm.atoms()
    assert set([2, a, b]) == set(atoms)


def test_atoms_nested():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')
    c = SingleVarFunctionScalar('c', 'x')

    sm = 2 + a + (b * c)

    print(sm)
    atoms = sm.atoms()
    assert set([2, a, b, c]) == set(atoms)


def test_sum_scalar():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')
    c = SingleVarFunctionScalar('c', 'x')
    d = SingleVarFunctionScalar('d', 'x')

    sm1 = a + b
    sm2 = c + d
    tot = sm1 + sm2
    assert len(tot) == 4
    assert set(tot) == set([a, b, c, d])

    tot += 2
    assert len(tot) == 5
    assert set(tot) == set([2, a, b, c, d])


def test_sum_vars():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')
    c = SingleVarFunctionScalar('c', 'y')

    sm = a + b + c
    assert get_variables(sm) == set(['x', 'y'])
    sm = replace_var(sm, 'y', 'x')
    assert get_variables(sm) == set(['x'])
    assert not has_variable(sm, 'y')


def test_setitem():
    a = SingleVarFunctionScalar('a', 'x')

    sm = 2 + a
    sm[1] = 1
    print(sm)
    sm = simplify(sm)

    assert sm == 3


def test_iszero_prod():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')

    prod = a * b
    prod[1] = 1e-17
    prod = simplify(prod)

    assert prod == 0
    assert is_zero(prod)


def test_sum_commutative():
    a = SingleVarFunctionScalar('a', 'x')
    b = SingleVarFunctionScalar('b', 'x')
    c = SingleVarFunctionScalar('c', 'x')
    d = SingleVarFunctionScalar('d', 'x')

    sm1 = a * b + c * d
    sm2 = d * c + b * a

    assert sm1 == sm2
