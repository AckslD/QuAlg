import numpy as np

from scalars import SingleVarFunctionScalar
from states import BaseQubitState
from fock_state import BaseFockState, FockOp, FockOpProduct
from operators import Operator, outer_product
from toolbox import simplify, replace_var, get_variables
from integrate import integrate

from scalars import ProductOfScalars, InnerProductFunction


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
    # bs = bscd
    # print(simplify(bs.inner_product(replace_var(bs))))
    phipsi = SingleVarFunctionScalar("phi", "w1") * SingleVarFunctionScalar("psi", "w2")
    sphipsi = (1 / 2) * phipsi * (bscc.to_state() + bsdc.to_state() - bscd.to_state() - bsdd.to_state())

    return s0, sphi, spsi, sphipsi


def construct_beam_splitter():
    fock_states = get_fock_states()
    qubit_states = [BaseQubitState(b).to_state() for b in ["00", "01", "10", "11"]]

    beam_splitter = sum((
        outer_product(fock_state, qubit_state)
        for fock_state, qubit_state in zip(fock_states, qubit_states)
    ), Operator())

    return beam_splitter.simplify()

def construct_projector(num_left, num_right):
    vac = BaseFockState([]).to_state()
    c1 = BaseFockState([FockOp('c', 'p1')]).to_state()
    c2 = BaseFockState([FockOp('c', 'p2')]).to_state()
    d1 = BaseFockState([FockOp('d', 'p1')]).to_state()
    d2 = BaseFockState([FockOp('d', 'p2')]).to_state()

    if (num_left, num_right) == (0, 0):
        # P00
        return outer_product(vac, vac)
    elif (num_left, num_right) == (1, 0):
        # P10
        return outer_product(c1, c1)
    elif (num_left, num_right) == (0, 1):
        # P10
        return outer_product(c1, c1)

def example_projectors():
    s0, sphi, spsi, sphipsi = get_fock_states()
    
    s = sphi
    p = construct_projector(1, 0)
    inner = (p * s).inner_product(s)
    print(integrate(inner))


def ultimate_example():
    u = construct_beam_splitter()
    p = construct_projector(1, 0)
    m = u.dagger() * p * replace_var(u)
    m = simplify(m)
    # print(m)

    s01 = BaseQubitState("10").to_state()
    inner = (m * s01).inner_product(s01)
    print(inner)
    print(integrate(inner))


def example_states():
    s0, sphi, spsi, sphipsi = get_fock_states()
    # print(f"State is: {sphi}\n")
    # inner = sphi.inner_product(sphi)
    # print(f"inner product is: {inner}\n")
    # simplify(inner)
    # print(f"after simplify: {simplify(inner)}\n")
    # inner = integrate(inner)
    # print(f"after integrate: {inner}\n")
    # return

    inner = simplify(sphipsi.inner_product(sphipsi))
    print(inner)
    print(integrate(inner))


def example_beam_splitter():
    beam_splitter = construct_beam_splitter()
    print(f"Beam splitter:\n{beam_splitter}")


def main():
    # example_states()
    # example_beam_splitter()
    # construct_projector(0, 0)
    # example_projectors()
    ultimate_example()


if __name__ == '__main__':
    main()
