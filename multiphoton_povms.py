from netsquid_ae.qdetector_multi import set_operators

import numpy as np
import math
from itertools import product

from scalars import SingleVarFunctionScalar, ComplexScalar
from states import BaseQubitState, State
from fock_state import BaseFockState, FockOp, FockOpProduct
from operators import Operator, outer_product
from toolbox import simplify, replace_var
from integrate import integrate


def get_fock_states(photon_number_a, photon_number_b):
    """Function that creates Fock states |n,m> for (n <= a) and (m <= b).

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

    """
    states = []
    for n in range(photon_number_a + 1):
        for m in range(photon_number_b + 1):

            a_n = BaseFockState().to_state()
            b_n = BaseFockState().to_state()

            for i in range(n):
                phi = SingleVarFunctionScalar(f"phi_{i}", f"w{i}")
                a_n @= phi * State(base_states=[BaseFockState([FockOp("c", f"w_{i}")]),
                                                BaseFockState([FockOp("d", f"w_{i}")])], scalars=[1, 1])

            for i in range(n, n+m):
                psi = SingleVarFunctionScalar(f"psi_{i}", f"w{i}")
                b_n @= psi * State(base_states=[BaseFockState([FockOp("c", f"w_{i}")]),
                                                BaseFockState([FockOp("d", f"w_{i}")])], scalars=[1, -1])

            norm = 1/np.sqrt(2**(n+m) * math.factorial(n) * math.factorial(m))
            state_nm = norm * a_n@b_n
            print("state |{},{}> = {}".format(n, m, simplify(state_nm)))

            # check length
            if len(state_nm) == 2**(n+m):
                print("has length:", len(state_nm))
            else:
                raise ValueError("State has the wrong length.")

            states.append(state_nm)

    return states


if __name__ == '__main__':
    # import operators with visibility=1 as a test
    kraus_ops, kraus_ops_num_res, outcome_dict = set_operators()

    states = get_fock_states(3, 3)
