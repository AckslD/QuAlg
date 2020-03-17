import pytest
import numpy as np

from qualg.toolbox import get_variables, replace_var, simplify
from qualg.q_state import BaseQubitState
from qualg.operators import outer_product, BaseOperator, Operator
from qualg.fock_state import BaseFockState, FockOp
from qualg.scalars import InnerProductFunction


def test_faulty_init_base_operator():
    with pytest.raises(TypeError):
        BaseOperator(None, None)


def test_eq():
    bs0 = BaseQubitState("0")
    bs1 = BaseQubitState("1")

    assert BaseOperator(bs0, bs0) == BaseOperator(bs0, bs0)
    assert BaseOperator(bs0, bs1) != BaseOperator(bs1, bs0)
    assert BaseOperator(bs0, bs0) != BaseOperator(bs1, bs1)


def test_non_complete_mul():
    bs0 = BaseQubitState("0")
    s00 = BaseQubitState("00").to_state()
    bop = BaseOperator(bs0, bs0)
    with pytest.raises(TypeError):
        bop * s00
    with pytest.raises(TypeError):
        bop * 1


def test_mul_state():
    z0 = BaseQubitState("0").to_state()
    z1 = BaseQubitState("1").to_state()
    x0 = (z0 + z1) * (1 / np.sqrt(2))
    x1 = (z0 - z1) * (1 / np.sqrt(2))
    H = outer_product(x0, z0) + outer_product(x1, z1)

    # Check the inner product after applying H
    assert np.isclose((H*z0).inner_product(z0), 1 / np.sqrt(2))
    assert np.isclose((H*z0).inner_product(z1), 1 / np.sqrt(2))
    assert np.isclose((H*z0).inner_product(x0), 1)
    assert np.isclose((H*z0).inner_product(x1), 0)

    assert np.isclose((H*x0).inner_product(z0), 1)
    assert np.isclose((H*x0).inner_product(z1), 0)
    assert np.isclose((H*x0).inner_product(x0), 1 / np.sqrt(2))
    assert np.isclose((H*x0).inner_product(x1), 1 / np.sqrt(2))


def test_to_numpy_matrix():
    # Only numbers
    phi = (1 / np.sqrt(2)) * (BaseQubitState("00").to_state() + BaseQubitState("11").to_state())
    op = outer_product(phi, phi)
    expected = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2
    print(op)
    print(op.to_numpy_matrix())
    assert np.all(np.isclose(op.to_numpy_matrix(), expected))

    # Symbolic scalars
    xy = InnerProductFunction("x", "y")
    phi = xy * (BaseQubitState("00").to_state() + BaseQubitState("11").to_state())
    op = outer_product(phi, phi)
    expected = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2

    def convert_scalars(scalar):
        return 1 / 2

    print(op)
    with pytest.raises(ValueError):
        op.to_numpy_matrix()
    m = op.to_numpy_matrix(convert_scalars=convert_scalars)
    print(m)
    assert np.all(np.isclose(m, expected))


def test_to_operator():
    bs0 = BaseQubitState("0")
    s0 = bs0.to_state()
    bop0 = BaseOperator(bs0, bs0)
    print(bop0.to_operator())
    print(Operator([bop0]))
    assert bop0.to_operator() == Operator([bop0])
    assert bop0.to_operator() == outer_product(s0, s0)


def test_fock_base_op():
    bsaw = BaseFockState([FockOp("a", "w")])
    bsav = BaseFockState([FockOp("a", "v")])
    bopaw = BaseOperator(bsaw, bsaw)
    bopav = BaseOperator(bsav, bsav)
    assert get_variables(bopaw) == set(["w"])
    assert get_variables(bopav) == set(["v"])
    new = replace_var(bopaw, "w", "v")
    assert bopav == new


def test_empty_operator():
    op = Operator()
    assert len(op) == 0


def test_faulty_init_operator():
    bs0 = BaseQubitState("0")
    bop0 = BaseOperator(bs0, bs0)
    with pytest.raises(ValueError):
        Operator([bop0], [1, 2])  # Wrong number of scalars

    with pytest.raises(TypeError):
        Operator([bs0], [1])  # Not a base operator

    with pytest.raises(TypeError):
        Operator([bop0], [bs0])  # Not a scalar

    bs00 = BaseQubitState("00")
    bop00 = BaseOperator(bs00, bs00)
    with pytest.raises(ValueError):
        Operator([bop0, bop00])  # Not add compatible


def test_mul_operator_scalar():
    bs0 = BaseQubitState("0")
    bop0 = BaseOperator(bs0, bs0)
    op = Operator([bop0])
    assert op * 0.5 == Operator([bop0], [0.5])


def test_mul_operator_operator():
    s0 = BaseQubitState("0").to_state()
    op0 = outer_product(s0, s0)
    assert op0 == op0 * op0


def test_mul_operator_state():
    s0 = BaseQubitState("0").to_state()
    s1 = BaseQubitState("1").to_state()
    splus = (s0 + s1) * (1 / np.sqrt(2))
    op0 = outer_product(s0, s0)
    assert s0 * (1 / np.sqrt(2)) == op0 * splus


def test_operator_dagger():
    # Y (hermitian)
    s0 = BaseQubitState("0").to_state()
    s1 = BaseQubitState("1").to_state()
    Y = outer_product(s1, s0) * 1j + outer_product(s0, s1) * (-1j)
    print(Y)
    assert Y == Y.dagger()

    # T (non-hermitian)
    s0 = BaseQubitState("0").to_state()
    s1 = BaseQubitState("1").to_state()
    T = outer_product(s0, s0) + outer_product(s1, s1) * np.exp(np.pi * 1j / 4)
    print(T)
    assert T != T.dagger()


def test_simplify_operator():
    bs0 = BaseQubitState("0")
    bs1 = BaseQubitState("1")
    op = Operator([BaseOperator(bs0, bs0)])
    op._terms[BaseOperator(bs1, bs1)] = 0

    assert len(op) == 2
    op = simplify(op)
    assert len(op) == 1


def test_operator_variables():
    bsaw = BaseFockState([FockOp("a", "w")])
    bsav = BaseFockState([FockOp("a", "v")])
    opaw = BaseOperator(bsaw, bsaw).to_operator()
    opav = BaseOperator(bsav, bsav).to_operator()
    assert get_variables(opaw) == set(["w"])
    assert get_variables(opav) == set(["v"])
    new = replace_var(opaw, "w", "v")
    assert opav == new
