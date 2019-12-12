# from netsquid_ae.qdetector_multi import set_operators

import numpy as np
import math
import time
import pickle
import multiprocessing


from scalars import SingleVarFunctionScalar, InnerProductFunction, ProductOfScalars, SumOfScalars, \
    is_number
from q_state import BaseQuditState
from states import State
from fock_state import BaseFockState, FockOp, FockOpProduct
from operators import Operator, outer_product
from toolbox import simplify, replace_var, get_variables
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

    #state = simplify(state_nm)

    return state_nm


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

    # replace variables
    num_vars = len(state.get_variables())
    w_j = []
    p_j = []
    for j in range(num_vars):
        w_j.append(f"w{j+1}")
        p_j.append(f"p{j+1}")
    for old, new in zip(w_j, p_j):
        state = replace_var(state, old, new)

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
    # TODO: do we really want different numbers from left and right?
    if num_photons_a != num_photons_b:
        print("Beam Splitter Unitary not symmertric.")

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
    qubit_states = [BaseQuditState(b, base=num_photons_a + 1).to_state() for b in combi]

    beam_splitter = sum((
        outer_product(fock_state, qubit_state)
        for fock_state, qubit_state in zip(fock_states, qubit_states)
    ), Operator())

    return beam_splitter.simplify()


def calculate_povm(clicks_left, clicks_right, max_num_photons_per_side):
    """Functions that calculate the effective POVM elements for (n, m) clicks (left, right) respectively.

    Note currently only working up to 9 incoming photons from each side, then qudit numbering fails.

    Parameters
    ----------
    clicks_left : int
        Number of clicks on the left.
    clicks_right : int
        Number of clicks on the right.
    max_num_photons_per_side : int
        Maximum number of photons incoming from each side.

    Returns
    -------
    m : instance of :class:'operators.Operator'
        Effective POVM M_i_j.

    """
    # TODO: fix this parameter
    incoming_photons = max_num_photons_per_side

    start_time = time.time()
    print(f"Generating M_{clicks_left}_{clicks_right}")

    u = construct_beam_splitter(incoming_photons, incoming_photons)
    p = construct_projector(clicks_left, clicks_right)
    # lmul unitary and projector
    m = u.dagger() * p
    m = simplify(m)
    # simplify
    for base_op, scalar in m._terms.items():
        scalar_variables = get_variables(scalar) - get_variables(base_op)
        m._terms[base_op] = integrate(scalar, scalar_variables)
    # rmul unitary
    m = m * replace_var(u)
    # simplify
    for base_op, scalar in m._terms.items():
        scalar_variables = get_variables(scalar) - get_variables(base_op)
        m._terms[base_op] = integrate(scalar, scalar_variables)
    '''# generate possible states
    combi = []
    for j in range(incoming_photons + 1):
        for k in range(incoming_photons + 1):
            combi.append(f"{j}" + f"{k}")
    states = [BaseQuditState(b, base=incoming_photons + 1).to_state() for b in combi]
    print("M_{}{}".format(clicks_left, clicks_right))
    # output individual matrix elements
    for sl, sr in product(states, repeat=2):
        inner = simplify(m * sr)
        inner = inner.inner_product(sl)
        bsl = next(iter(sl))[0]
        bsr = next(iter(sr))[0]
        print(f"\t{bsl}{bsr._bra_str()}: {integrate(inner)}")'''

    print(f"Generating M_{clicks_left}_{clicks_right} took {time.time() - start_time}")

    return simplify(m)


def wrap_povm(left, right, tot):
    """Short wrapper to return tuple of (parameters, result) in multiprocessing."""
    return (left, right), calculate_povm(left, right, tot)


def generate_effective_povms(incoming_left, incoming_right, subset=None):
    """Function that generates all possible POVM operators for arbitrary number of incoming photons from the left
    and the right.

    Note: Should currently only be used with up to 9 from each side, because 2-digit number mess up the naming of the
    QuditStates.

    Parameters
    ----------
    incoming_left : int
        Maximum number of incoming photons from the left.
    incoming_right : int
        Maximum number of incoming photons from the right.
    subset : None, "leq" or "g"
        Mode of slicing the full set of POVM into subset

    Returns
    -------
    operators : dict of operators (:class:'operators.Operator') and tuples as keys
        Dictionary of all possible POVM operators for the given number of incoming photons.

    """
    total_photon_number = incoming_left + incoming_right
    operators = {}
    arguments = []
    for n in range(total_photon_number + 1):
        for m in range(total_photon_number + 1):
            if n + m <= total_photon_number:
                if subset == "leq":
                    if n <= m:
                        arguments.append((n, m, max(incoming_left, incoming_right)))
                elif subset == "g":
                    if n > m:
                        arguments.append((n, m, max(incoming_left, incoming_right)))
                elif subset is None:
                    arguments.append((n, m, max(incoming_left, incoming_right)))
                else:
                    raise ValueError(f"subset should be None, 'leq' or 'g' and not {subset}.")

    with multiprocessing.Pool() as pool:
        results = pool.starmap_async(wrap_povm, arguments)
        for op in results.get():
            operators[op[0]] = op[1]

    return operators


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


if __name__ == '__main__':
    # import operators with visibility=1 as a test
    # kraus_ops, kraus_ops_num_res, outcome_dict = set_operators()
    num = 3

    '''s, s_dict = generate_fock_states(num, num)
    print("|{},{}> = {}".format(3, 3, s_dict[3, 3]))
    p, p_dict = generate_projectors(2*num)
    print()
    print(len(s))
    print("P_{}_{}: {}".format(2, 4, p_dict[(2, 4)]))
    construct_beam_splitter(num, num)'''

    '''start_time = time.time()
    n, m, tot = 0, 4, 3
    print(f"Generating M_{n}_{m}")
    povm = calculate_povm(n, m, tot)
    middle_time = time.time()
    array = povm.to_numpy_matrix(convert_scalars)
    end_time = time.time()
    print(povm)
    print(array)
    print(f"Time elapsed {end_time - start_time} (middle {middle_time - start_time})")
    exit()'''

    start_time = time.time()
    # m = calculate_povm(0, 0, 3)
    subset = "g"
    povms = generate_effective_povms(3, 3, subset=subset)
    print(f"Time elapsed {time.time() - start_time}")
    with open(f'multiphoton_povms_raw_subset_{subset}.pkl', 'wb') as output:
        pickle.dump(povms, output, pickle.HIGHEST_PROTOCOL)
    exit()
    '''print("elapsed time:", time.time() - start_time)
    gen_time = time.time()
    # print(povms.to_numpy_matrix(convert_scalars))
    arrays = []
    for p in povms:
        arrays.append(p.to_numpy_matrix(convert_scalars))
    print(f"conversion took {time.time()-gen_time}")
    with open('multiphoton_povms_arrays_full.pkl', 'wb') as output:
        pickle.dump(arrays, output, pickle.HIGHEST_PROTOCOL)'''

