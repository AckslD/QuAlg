"""
This example shows how the POVMs used in the paper https://arxiv.org/abs/1903.09778 can be computed
using QuAlg.
"""
import numpy as np
from timeit import default_timer as timer

from qualg.scalars import SingleVarFunctionScalar, InnerProductFunction, ProductOfScalars,\
    SumOfScalars, is_number
from qualg.q_state import BaseQubitState
from qualg.fock_state import BaseFockState, FockOp
from qualg.operators import Operator, outer_product
from qualg.toolbox import simplify, replace_var
from qualg.integrate import integrate


def get_fock_states():
    """Construct the fock states (eq. (42)-(45) in https://arxiv.org/abs/1903.09778)"""
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
    sphipsi = (1 / 2) * phipsi * (bscc.to_state() + bsdc.to_state() - bscd.to_state() - bsdd.to_state())

    return s0, sphi, spsi, sphipsi


def construct_beam_splitter():
    """Construct the beam splitter operation (eq. (40)-(41) in https://arxiv.org/abs/1903.09778)"""
    fock_states = list(get_fock_states())
    for i, state in enumerate(fock_states):
        for old, new in zip(['w1', 'w2'], ['b1', 'b2']):
            state = replace_var(state, old, new)
        fock_states[i] = state
    qubit_states = [BaseQubitState(b).to_state() for b in ["00", "01", "10", "11"]]

    beam_splitter = sum((
        outer_product(fock_state, qubit_state)
        for fock_state, qubit_state in zip(fock_states, qubit_states)
    ), Operator())

    return beam_splitter.simplify()


def construct_projector(num_left, num_right):
    """Construct the projector (eq. (46)-(51) in https://arxiv.org/abs/1903.09778)"""
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
        # P01
        return outer_product(d1, d1)
    elif (num_left, num_right) == (1, 1):
        # P11
        return outer_product(c1@d2, c1@d2)
    elif (num_left, num_right) == (2, 0):
        # P20
        return outer_product(c1@c2, c1@c2)
    elif (num_left, num_right) == (0, 2):
        # P02
        return outer_product(d1@d2, d1@d2)
    else:
        raise NotImplementedError()


def example_states(no_output=False):
    """Example showing how the states can be constructed"""
    s0, sphi, spsi, sphipsi = get_fock_states()
    if not no_output:
        print(f"State is: {sphi}\n")
    inner = sphi.inner_product(sphi)
    if not no_output:
        print(f"inner product is: {inner}\n")
    simplify(inner)
    if not no_output:
        print(f"after simplify: {simplify(inner)}\n")
    inner = integrate(inner)
    if not no_output:
        print(f"after integrate: {inner}\n")
    return

    inner = simplify(sphipsi.inner_product(sphipsi))
    if not no_output:
        print(inner)
        print(integrate(inner))


def example_beam_splitter(no_output=False):
    """Example showing how the beam spliter operation can be constructed"""
    beam_splitter = construct_beam_splitter()
    if not no_output:
        print(f"Beam splitter:\n{beam_splitter}")


def example_projectors(no_output=False):
    """Example showing how to constuct the projectors and how they act on a given state."""
    s0, sphi, spsi, sphipsi = get_fock_states()

    s = sphi
    p = construct_projector(1, 0)
    inner = (p * s).inner_product(s)
    if not no_output:
        print(integrate(inner))


def ultimate_example(indices, no_output=False):
    """Example for computing one of the POVMs (eq. (82) - (87) of https://arxiv.org/abs/1903.09778)"""
    def convert_scalars(scalar):
        visibility = 0.9

        scalar = integrate(scalar)
        if is_number(scalar):
            return scalar
        for sequenced_class in [ProductOfScalars, SumOfScalars]:
            if isinstance(scalar, sequenced_class):
                return simplify(sequenced_class([convert_scalars(s) for s in scalar]))
        if isinstance(scalar, InnerProductFunction):
            if set(scalar._func_names) == set(['phi', 'psi']):
                return visibility
        raise RuntimeError(f"unknown scalar {scalar} of type {type(scalar)}")

    assert len(indices) == 2

    u = construct_beam_splitter()
    p = construct_projector(*indices)
    m = u.dagger() * p * replace_var(u)
    m = simplify(m)
    if not no_output:
        # print(m)
        print(m.to_numpy_matrix(convert_scalars))


def main(no_output=False):
    t1 = timer()
    # example_states(no_output=no_output)
    # example_beam_splitter(no_output=no_output)
    # example_projectors(no_output=no_output)
    ultimate_example((1, 0), no_output=no_output)
    t2 = timer()
    if not no_output:
        print(f"\nTime: {t2 - t1}")


if __name__ == '__main__':
    main()
