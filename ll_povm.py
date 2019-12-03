import numpy as np

from scalars import SingleVarFunctionScalar
from states import BaseQubitState
from fock_state import BaseFockState, FockOp, FockOpProduct
from operators import Operator, outer_product
from toolbox import simplify, replace_var
from integrate import integrate


def construct_beam_splitter():
    # c = FockOp("c", "w")
    # d = FockOp("d", "w")
    # fop = FockOpProduct([c, d])
    # print(fop)
    # print(fop._key())
    # print(hash(fop))
    # return

    bs0 = BaseFockState()
    bsc = BaseFockState([FockOp("c", "w")])
    bsd = BaseFockState([FockOp("d", "w")])
    phi = SingleVarFunctionScalar("phi", "w")
    psi = SingleVarFunctionScalar("psi", "w")

    s0 = bs0.to_state()
    f = 1 / np.sqrt(2)
    sphi = f * phi * (bsc.to_state() + bsd.to_state())
    # print(f"State:\n{sphi}\n")
    spsi = f * psi * (bsc.to_state() - bsd.to_state())
    sphipsi = psi * (bsc.to_state() - bsd.to_state())

    # print(s0.inner_product(s0))
    # print(replace_var(sphi, 'w', 'v'))
    # print()
    inner = sphi.inner_product(replace_var(sphi, 'w', 'v'))
    # print(inner)
    # print(simplify(inner))
    # print(integrate(simplify(inner), 'w'))
    # print(integrate(inner, 'w'))
    i = simplify(integrate(inner, 'w'))
    i = simplify(i)
    print(i)
    import pdb
    pdb.set_trace()
    ii = integrate(i, 'v')
    print(ii)
    print(simplify(ii))
    # print(simplify(i))
    # for ss in i:
    #     for s in ss:
    #         print(f"{s} : {type(s)}")
    #     print()
        # print(ss)
    return
    # print(integrate(simplify(sphi.inner_product(sphi.replace_var('w', 'v')), 'w'))

    fock_states = [s0, spsi, sphi, sphipsi]
    qubit_states = [BaseQubitState(b).to_state() for b in ["00", "01", "10", "11"]]

    beam_splitter = sum((
        outer_product(fock_state, qubit_state)
        for fock_state, qubit_state in zip(fock_states, qubit_states)
    ), Operator())

    return beam_splitter.simplify()


beam_splitter = construct_beam_splitter()
# print(f"Beam splitter:\n{beam_splitter}")
