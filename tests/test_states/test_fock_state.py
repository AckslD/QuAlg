import pytest

from qualg.toolbox import simplify
from qualg.scalars import DeltaFunction
from qualg.fock_state import FockOp, FockOpProduct, BaseFockState


def test_fock_op_product():
    aw = FockOp('a', 'w')
    av = FockOp('a', 'v')

    print(FockOpProduct([aw, av]))
    print(FockOpProduct([aw, av, aw]))

    assert FockOpProduct([aw, av]) == FockOpProduct([av, aw])
    assert FockOpProduct([aw, av, aw]) == FockOpProduct([av, aw, aw])


def test_inner_product():
    aw = FockOp('a', 'w')
    av = FockOp('a', 'v')
    cw = FockOp('c', 'w')
    saw = BaseFockState([aw])
    sav = BaseFockState([av])
    scw = BaseFockState([cw])
    assert simplify(saw.inner_product(sav)) == DeltaFunction('w', 'v')
    assert simplify(saw.inner_product(scw)) == 0


def test_inner_product_multi_photon():
    cw1 = FockOp('c', 'w1')
    cw2 = FockOp('c', 'w2')
    cw3 = FockOp('c', 'w3')
    state = BaseFockState([cw1, cw2, cw3]).to_state()
    print(state)
    print()
    inp = simplify(state.inner_product(state))
    print(inp)
    assert len(inp) == 6
    print(inp[0])
    assert len(inp[0]) == 3


@pytest.mark.parametrize("op1, op2, expected", [
    (FockOp("a", "w"), FockOp("a", "w"), True),
    (FockOp("a", "w"), FockOp("a", "w", False), False),
    (FockOp("a", "w"), FockOp("a", "v"), False),
    (FockOp("a", "w"), FockOp("b", "w"), False),
    (FockOp("a", "w"), FockOp("b", "v"), False),
    (FockOp("a", "w"), "a(w)", False),
])
def test_fock_op_eq(op1, op2, expected):
    assert (op1 == op2) == expected


def test_fock_op_dagger():
    op1 = FockOp("a", "w")
    op2 = FockOp("a", "w", creation=False)
    assert op1 != op2
    op2 = op2.dagger()
    assert op1 == op2


@pytest.mark.parametrize("op1, op2, expected", [
    (
        FockOpProduct([FockOp('a', 'w')]),
        FockOp('a', 'w'),
        FockOpProduct([FockOp('a', 'w'), FockOp('a', 'w')]),
    ),
    (
        FockOpProduct([FockOp('a', 'w')]),
        FockOpProduct([FockOp('a', 'w')]),
        FockOpProduct([FockOp('a', 'w'), FockOp('a', 'w')]),
    ),
    (
        FockOpProduct([FockOp('a', 'w')]),
        FockOpProduct([FockOp('b', 'v')]),
        FockOpProduct([FockOp('a', 'w'), FockOp('b', 'v')]),
    ),
])
def test_fock_op_product_mul(op1, op2, expected):
    prod = op1 * op2
    print(prod)
    assert prod == expected


@pytest.mark.parametrize("op1, op2, expected", [
    (FockOpProduct([FockOp("a", "w")]), FockOpProduct([FockOp("a", "w")]), True),
    (FockOpProduct([
        FockOp("a", "w"),
        FockOp("a", "w")
    ],), FockOpProduct([
        FockOp("a", "w"),
        FockOp("a", "w")
    ]), True),
    (FockOpProduct([
        FockOp("a", "w"),
        FockOp("a", "w")
    ],), FockOpProduct([
        FockOp("a", "w"),
        FockOp("a", "v")
    ]), False),
    (FockOpProduct([
        FockOp("a", "w"),
        FockOp("a", "w")
    ],), FockOpProduct([
        FockOp("a", "w"),
    ]), False),
    (FockOpProduct([FockOp("a", "w")]), "a", False),
])
def test_fock_op_product_eq(op1, op2, expected):
    assert (op1 == op2) == expected
    if isinstance(op2, FockOpProduct):
        assert (BaseFockState(op1) == BaseFockState(op2)) == expected
    else:
        assert (BaseFockState(op1) == op2) == expected


def test_fock_op_product_variables_in_mode():
    op = FockOpProduct([
        FockOp("a", "w"),
        FockOp("a", "v"),
        FockOp("b", "x"),
    ])
    assert set(op.variables_in_mode("a")) == set(["w", "v"])
    assert set(op.variables_in_mode("b")) == set(["x"])

    output_dict = op.variables_by_modes()
    expected_dict = {"a": ["w", "v"], "b": ["x"]}
    assert len(output_dict) == len(expected_dict)
    for key in output_dict.keys():
        assert set(output_dict[key]) == set(expected_dict[key])


def test_fock_state_shape():
    bs = BaseFockState([FockOp("a", "w")])
    assert bs.shape is None


def test_fock_state_tensor_product():
    fock_prod1 = FockOpProduct([FockOp("a", "w")])
    fock_prod2 = FockOpProduct([FockOp("a", "w"), FockOp("a", "v")])
    bs1 = BaseFockState(fock_prod1)
    bs2 = BaseFockState(fock_prod2)

    assert bs1 @ bs2 == BaseFockState(fock_prod1 * fock_prod2)
