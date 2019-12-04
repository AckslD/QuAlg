import numpy as np

from scalars import SingleVarFunctionScalar
from states import BaseQubitState
from fock_state import BaseFockState, FockOp, FockOpProduct
from operators import Operator, outer_product
from toolbox import simplify, replace_var
from integrate import integrate


def get_fock_states():
    bs0 = BaseFockState()
    bsc = BaseFockState([FockOp("c", "w1")])
    bsd = BaseFockState([FockOp("d", "w1")])
    phi = SingleVarFunctionScalar("phi", "w1")
    psi = SingleVarFunctionScalar("psi", "w1")

    s0 = bs0.to_state()
    f = 1 / np.sqrt(2)
    sphi = f * phi * (bsc.to_state() + bsd.to_state())
    spsi = f * psi * (bsc.to_state() - bsd.to_state())

    bscc = BaseFockState([FockOp("c", "w1"), FockOp("c", "w2")])
    bscd = BaseFockState([FockOp("c", "w1"), FockOp("d", "w2")])
    bsdc = BaseFockState([FockOp("d", "w1"), FockOp("c", "w2")])
    bsdd = BaseFockState([FockOp("d", "w1"), FockOp("d", "w2")])
    phipsi = SingleVarFunctionScalar("phi", "w1") * SingleVarFunctionScalar("psi", "w2")
    sphipsi = phipsi * (bscc.to_state() + bsdc.to_state() + bscd.to_state() - bsdd.to_state())

    return s0, sphi, spsi, sphipsi


def construct_beam_splitter():
    fock_states = get_fock_states()
    qubit_states = [BaseQubitState(b).to_state() for b in ["00", "01", "10", "11"]]

    beam_splitter = sum((
        outer_product(fock_state, qubit_state)
        for fock_state, qubit_state in zip(fock_states, qubit_states)
    ), Operator())

    return beam_splitter.simplify()


def example_states():
    s0, sphi, spsi, sphipsi = get_fock_states()
    print(f"State is: {sphi}\n")
    inner = sphi.inner_product(replace_var(sphi, 'w1', 'v1'))
    print(f"inner product is: {inner}\n")
    print(f"after simplify: {simplify(inner)}\n")
    inner = integrate(inner, ['w1', 'v1'])
    print(f"after integrate: {inner}\n")
    return


    r = replace_var(replace_var(sphipsi, 'w1', 'v1'), 'w2', 'v2')
    l = sphipsi

    i = simplify(r.inner_product(l))
    print(i)
    # print()
    # print(integrate(i, 'w1'))
    # print()
    # print(simplify(integrate(i, 'w1')))
    # print()
    # print(integrate(simplify(integrate(i, 'w1')), 'w2'))
    print(integrate(i, ['w1', 'w2', 'v1', 'v2']))


def example_beam_splitter():
    beam_splitter = construct_beam_splitter()
    print(f"Beam splitter:\n{beam_splitter}")


def main():
    example_states()
    # example_beam_splitter()


if __name__ == '__main__':
    main()