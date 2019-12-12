import pickle
import time

from multiphoton_povms import convert_scalars
from operators import Operator, BaseOperator
from q_state import BaseQuditState
from scalars import ProductOfScalars, InnerProductFunction, SumOfScalars

from netsquid.qubits import operators as ops

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

    output_file = open("subset_povms_3_3_photons.txt", "w")
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
    # open text file
    operator_dict = read_from_txt("operators.txt")
    # convert dict to arrays with given visibility
    for key in operator_dict.keys():

    # add operators
    # return
        pass


if __name__ == '__main__':
    write_to_txt('multiphoton_povms_raw_2.pkl')
    exit()
    with open("multiphoton_povms_arrays_4.pkl", "rb") as file:
        arrays = pickle.load(file)
    with open("multiphoton_povms_raw_4.pkl", "rb") as file:
        operators = pickle.load(file)

    total_photon_number = 4
    names = []
    for n in range(total_photon_number + 1):
        for m in range(total_photon_number + 1):
            if n + m <= total_photon_number:
                names.append((n, m))
    array_dict = {}
    ops_dict = {}
    for i in range(len(arrays)):
        array_dict[names[i]] = arrays[i]
        ops_dict[names[i]] = operators[i]
    with open("multiphoton_povms_dict_4.pkl", "wb") as file:
        pickle.dump(ops_dict, file)

    write_to_txt("multiphoton_povms_dict_4.pkl")
    op_dict = read_from_txt("operators.txt")
    print(op_dict.keys())

    start_time = time.time()
    for o in operators:
        o.to_numpy_matrix(convert_scalars)
    print(f"elapsed time for conversion: {time.time()-start_time} s")

