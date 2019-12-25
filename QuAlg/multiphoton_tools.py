import pickle
import time
import numpy as np
import os

from QuAlg.multiphoton_povms import convert_scalars
from QuAlg.operators import Operator, BaseOperator
from QuAlg.q_state import BaseQuditState
from QuAlg.scalars import ProductOfScalars, InnerProductFunction, SumOfScalars
from QuAlg.toolbox import simplify, get_variables
from QuAlg.integrate import integrate

from netsquid.qubits import operators as ops
from scipy.linalg import sqrtm


def write_to_txt(filename, names=None):
    """Function that writes pre-generated files to a text file 'operators.txt'.

    Parameters
    ----------
    filename : str
        Name of the file containing the QuAlg operators dictionary or list.
    names : list
        List of names corresponding to the operators in the list.

    """
    with open(filename, "rb") as file:
        ops_dict = pickle.load(file)

    output_file = open("full_multiphoton_povms.txt", "w")
    if isinstance(ops_dict, dict):
        for k in ops_dict.keys():
            output_file.write(f"{k} \n")
            output_file.write(repr(ops_dict[k]) + " \n")
    elif isinstance(ops_dict, list):
        for op in ops_dict:
            output_file.write(repr(op) + " \n")
    output_file.close()


def read_from_txt(filename):
    """Function that reads operators from a text file.

    Parameters
    ----------
    filename : str
        Name of the text file containing the operator representations.

    Returns
    -------
    operator_dict : dict
        Dictionary containing the Operators and their keys (n,m)

    """
    keys = []
    ops = []
    with open(filename) as fp:
        line = fp.readline()
        n = int(line.strip()[1])
        m = int(line.strip()[4])
        keys.append((n, m))
        cnt = 1
        while line:
            line = fp.readline()
            if cnt % 2 == 1:
                ops.append(line.strip())
            else:
                if len(line.strip()) != 0:
                    n = int(line.strip()[1])
                    m = int(line.strip()[4])
                    keys.append((n, m))
            cnt += 1
    ops_dict = {}
    for i in range(len(keys)):
        ops_dict[keys[i]] = eval(ops[i])

    return ops_dict


def set_operators(visibility):
    """Computes the Kraus operators for both threshold and number resolving photon detectors.

    Parameters
    ----------
    visibility : float
        Visibility / photon indistinguishability at the BSM beam splitter.

    Returns
    -------
    kraus_operators : list
        List containing the Kraus operators for a NON-photon-number-resolving BSM.
    kraus_operators_num_res : list
        List containing the Kraus operators for a photon-number-resolving BSM.
    outcome_dict : dict
        Dictionary with keys: int(list index) and values: tuple(operator name).

    """
    # open text file
    operator_dict = read_from_txt("netsquid_ae/full_multiphoton_povms.txt")

    kraus_operators_num_res = []
    outcome_dict = {}
    idx = 0
    P_01 = np.zeros((16, 16), dtype=complex)
    P_10 = np.zeros((16, 16), dtype=complex)
    P_11 = np.zeros((16, 16), dtype=complex)
    for key in operator_dict.keys():
        # convert dict to arrays with given visibility
        array = sqrtm(operator_dict[key].to_numpy_matrix(convert_scalars, visibility))
        kraus_operators_num_res.append(ops.Operator("n_{}{}".format(key[0], key[1]), array))
        outcome_dict[idx] = key
        idx += 1
        # Add up operators with same click pattern for non-number-resolving case
        (n, m) = key
        if key == (0, 0):
            P_00 = array
        elif n == 0 and m > 0:
            P_01 += array
        elif n > 0 and m == 0:
            P_10 += array
        elif n > 0 and m > 0:
            P_11 += array

# Take the matrix square root of the POVM to get the Kraus operator
    n_00 = ops.Operator("n_00", P_00)
    n_01 = ops.Operator("n_01", P_01)
    n_10 = ops.Operator("n_10", P_10)
    n_11 = ops.Operator("n_11", P_11)
    kraus_operators = [n_00, n_01, n_10, n_11]
    # return
    return kraus_operators, kraus_operators_num_res, outcome_dict


if __name__ == '__main__':
    with open("multiphoton_povms_raw_subset_leq.pkl", "rb") as file:
        leq_dict = pickle.load(file)
    with open("multiphoton_povms_raw_subset_g.pkl", "rb") as file2:
        g_dict = pickle.load(file2)
    leq_dict.update(g_dict)
    with open('full_multiphoton_povms.pkl', 'wb') as output:
        pickle.dump(leq_dict, output, pickle.HIGHEST_PROTOCOL)
    write_to_txt("full_multiphoton_povms.pkl")
    exit()

    write_to_txt('multiphoton_povms_raw_full_2_2.pkl')
    with open("multiphoton_povms_raw_subset_g.pkl", "rb") as file:
        ops_dict = pickle.load(file)
    print(len(ops_dict.keys()))
    for key in ops_dict.keys():
        print(key)
        print(ops_dict[key])
        print()
    exit()
    total_photon_number = 6
    keys = []
    for n in range(total_photon_number + 1):
        for m in range(total_photon_number + 1):
            if n + m <= total_photon_number:
                keys.append((n, m))

    with open("multiphoton_povms_raw_full.pkl", "rb") as file:
        ops_raw = pickle.load(file)
    ops_simplified = []
    ops_dict = {}
    for i in range(len(ops_raw)):
        m = ops_raw[i]
        start_time = time.time()
        print(f"Simplifying {keys[i]}")
        for base_op, scalar in m._terms.items():
            scalar_variables = get_variables(scalar) - get_variables(base_op)
            m._terms[base_op] = integrate(scalar, scalar_variables)
        m = simplify(m)
        ops_simplified.append(m)
        ops_dict[keys[i]] = m
        print(f"Simplifying {keys[i]} took {time.time()-start_time}s.")
    with open(f'simp_povms_full_3_3.pkl', 'wb') as output:
        pickle.dump(ops_dict, output, pickle.HIGHEST_PROTOCOL)
