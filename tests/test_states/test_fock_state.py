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
