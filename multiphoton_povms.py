from netsquid_ae.qdetector_multi import set_operators

import numpy as np
import math
from itertools import product
import time


from scalars import SingleVarFunctionScalar
from q_state import BaseQubitState
from states import State
from fock_state import BaseFockState, FockOp, FockOpProduct
from operators import Operator, outer_product
from toolbox import simplify, replace_var
from integrate import integrate


def generate_fock_states(photon_number_a, photon_number_b):
    """Function that generates Fock states |n,m> for (n <= a) and (m <= b).

    Parameters
    ----------
    photon_number_a : int
        Maximum number of photons in mode a.
    photon_number_b : int
        Maximum number of photons in mode b.

    Returns
    -------
    states : list
        List of all Fock states |n,m> with (n <= a) and (m <= b).
    states : dict
        Dictionary with state names (key) and states (value).

    """
    states = []
    states_dict = {}
    for n in range(photon_number_a + 1):
        for m in range(photon_number_b + 1):

            state = construct_fock_state(n, m)
            states.append(state)
            states_dict[(n, m)] = state

    return states, states_dict


def construct_fock_state(num_mode_a, num_mode_b):
    """Constructs the Fock state |n,m> for arbitrary photon numbers in mode a and mode b.

    Parameters
    ----------
    num_mode_a : int
        Number of photons in mode a.
    num_mode_b : int
        Number of photons in mode b.

    Returns
    -------
    state : instance of :class:`states.State`
        Fock state |n,m>.

    """
    a_n = BaseFockState().to_state()
    b_n = BaseFockState().to_state()

    for i in range(num_mode_a):
        #phi = SingleVarFunctionScalar(f"phi_{i+1}", f"w{i}")
        phi = SingleVarFunctionScalar(f"phi", f"w{i}")
        a_n @= phi * State(base_states=[BaseFockState([FockOp("c", f"w{i}")]),
                                        BaseFockState([FockOp("d", f"w{i}")])], scalars=[1, 1])

    for i in range(num_mode_a, num_mode_a+num_mode_b):
        #psi = SingleVarFunctionScalar(f"psi_{i-num_mode_a+1}", f"w{i}")
        psi = SingleVarFunctionScalar(f"psi", f"w{i}")
        b_n @= psi * State(base_states=[BaseFockState([FockOp("c", f"w{i}")]),
                                        BaseFockState([FockOp("d", f"w{i}")])], scalars=[1, -1])

    norm = float(1/np.sqrt(2**(num_mode_a+num_mode_b) * math.factorial(num_mode_a) * math.factorial(num_mode_b)))

    state_nm = simplify(a_n@b_n)
    state_nm = norm * state_nm
    # print("state |{},{}> = {}".format(num_mode_a, num_mode_b, simplify(state_nm)))

    # check length
    if len(state_nm) != 2**(num_mode_a+num_mode_b):
        raise ValueError("State has the wrong length.")

    state = simplify(state_nm)

    return state


def generate_projectors(max_number_photons):
    """Generate all photon number projectors P_i,j on photon numbers up to n = max_num_photons.

    Parameters
    ----------
    max_number_photons : int
        Maximum number of photons i + j to make combinations for.

    Returns
    -------
    projectors : list
        List of all possible Projectors P_i_j with i+j <= max_number_photons.
    projectors_dict : dict
        Dictionary with all tuples (n,m) and their corresponding projector P_i_j.

    """
    # compute all possible combinations of i,j
    combinations = []
    for n in range(max_number_photons+1):
        for m in range(max_number_photons+1):
            if n + m <= max_number_photons:
                combinations.append((n, m))
    # generate the corresponding projectors
    projectors = []
    projectors_dict = {}
    for (n, m) in combinations:
        proj = construct_projector(n, m)
        projectors.append(proj)
        projectors_dict[(n, m)] = proj
        # print("P_{}_{} is : {}".format(n, m, construct_projector(n, m)))

    return projectors, projectors_dict


def construct_projector(num_left, num_right):
    """Constructs the projectors on photon numbers P_{num_left}_{num_right}.

    Parameters
    ----------
    num_left : int
        Index i of the desired Projector P_i_j.
    num_right : int
        Index j of the desired Projector P_i_j.

    Returns
    -------
    p_i_j : instance of :class:'operators.Operator'
        Projector P_i_j on photon numbers i on the left and j on the right.

    """
    # generate the state from Fock operators
    state = BaseFockState().to_state()
    for i in range(num_left):
        state @= BaseFockState([FockOp("c", f"w{i+1}")]).to_state()
    for i in range(num_left, num_left+num_right):
        state @= BaseFockState([FockOp("d", f"w{i+1}")]).to_state()
    # state = simplify(state)
    norm = float(1/np.sqrt(math.factorial(num_left) * math.factorial(num_right)))
    state = norm * state
    p_i_j = outer_product(state, state)
    return p_i_j


def construct_beam_splitter(num_photons_a, num_photons_b):
    """Function constructing the beam splitter Unitary for arbitrary number of incoming photons.

    Parameters
    ----------
    num_photons_a : int
        Number of Photons incoming from the left.
    num_photons_b : int
        Number of Photons incoming from the right.

    Returns
    -------
    beam_splitter : instance of :class:'operators.Operator'
        Unitary describing the action of the beam splitter.

    """
    fock_states, _ = generate_fock_states(num_photons_a, num_photons_b)
    # replace variables
    for i, state in enumerate(fock_states):
        num_vars = len(state.get_variables())
        w_j = []
        b_j = []
        for j in range(num_vars):
            w_j.append(f"w{j+1}")
            b_j.append(f"b{j+1}")
        for old, new in zip(w_j, b_j):
            state = replace_var(state, old, new)
        fock_states[i] = state
    # generate qubit states
    combi = []
    for n in range(num_photons_a + 1):
        for m in range(num_photons_b + 1):
            combi.append(f"{n}" + f"{m}")
    qubit_states = [BaseQuditState(b).to_state() for b in combi]

    beam_splitter = sum((
        outer_product(fock_state, qubit_state)
        for fock_state, qubit_state in zip(fock_states, qubit_states)
    ), Operator())

    return beam_splitter.simplify()


def calculate_povm(clicks_left, clicks_right):
    """Functions that calculate the effective POVM elements for (n, m) clicks (left, right) respectively.

    Note currently only working up to 9 incoming photons from each side, then qudit numbering fails.

    Parameters
    ----------
    clicks_left : int
        Maximum number of photons coming from the left.
    clicks_right : int
        Maximum number of photons coming from the right.

    Returns
    -------
    m : instance of :class:'operators.Operator'
        Effective POVM M_i_j.

    """
    incoming_photons = 3

    u = construct_beam_splitter(incoming_photons, incoming_photons)
    p = construct_projector(clicks_left, clicks_right)
    m = u.dagger() * p * replace_var(u)
    m = simplify(m)
    print(m)
    # generate qubit states
    combi = []
    for j in range(incoming_photons + 1):
        for k in range(incoming_photons + 1):
            combi.append(f"{j}" + f"{k}")
    states = [BaseQuditState(b).to_state() for b in combi]
    print("M_{}{}".format(clicks_left, clicks_right))
    for sl, sr in product(states, repeat=2):
        inner = (m * sr).inner_product(sl)
        bsl = next(iter(sl))[0]
        bsr = next(iter(sr))[0]
        print(f"\t{bsl}{bsr._bra_str()}: {integrate(inner)}")
    print(type(m))
    return m


if __name__ == '__main__':
    # import operators with visibility=1 as a test
    kraus_ops, kraus_ops_num_res, outcome_dict = set_operators()

    num = 3

    '''s, s_dict = generate_fock_states(num, num)
    print("|{},{}> = {}".format(3, 3, s_dict[3, 3]))
    p, p_dict = generate_projectors(2*num)
    print()
    print(len(s))
    print("P_{}_{}: {}".format(2, 4, p_dict[(2, 4)]))
    construct_beam_splitter(num, num)'''
    start_time = time.time()
    calculate_povm(3, 3)
    print("elapsed time:", time.time() - start_time)

