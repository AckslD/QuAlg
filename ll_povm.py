from scalars import SingleVarFunctionScalar
from states import BaseQubitState
from fock_state import BaseFockState, FockOp, FockOpProduct
from operators import Operator, outer_product


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
    sphi = phi * (bsc.to_state() + bsd.to_state())
    spsi = psi * (bsc.to_state() - bsd.to_state())
    sphipsi = psi * (bsc.to_state() - bsd.to_state())

    fock_states = [s0, spsi, sphi, sphipsi]
    qubit_states = [BaseQubitState(b).to_state() for b in ["00", "01", "10", "11"]]

    return sum((
        outer_product(fock_state, qubit_state)
        for fock_state, qubit_state in zip(fock_states, qubit_states)
    ), Operator())

    print(sphi)
    print(spsi)


beam_splitter = construct_beam_splitter()
print(beam_splitter)
print()
print(beam_splitter.simplify())
