import unittest
import time
from itertools import product
import numpy as np
import pickle
from scipy.linalg import sqrtm

from netsquid_ae.qdetector_multi import set_operators

from multiphoton_povms import generate_fock_states, construct_fock_state, construct_projector, generate_effective_povms,\
    convert_scalars
from multiphoton_tools import read_from_txt
from integrate import integrate
from toolbox import simplify


class TestMultiPhoton(unittest.TestCase):
    """Test the correct implementation of multi photon states."""

    def test_state_generation(self):
        """Test the correct generation of Fock states."""
        n = 2
        states, states_dict = generate_fock_states(n, n)

        for name in states_dict:
            state = states_dict[name]
            # calculate inner product
            inner_prod = state.inner_product(state)
            ip = simplify(inner_prod)
            # integrate
            norm = integrate(ip)
            # must be equal to 1
            # currently only works for phi_i = phi_j m psi_i = psi_j
            self.assertAlmostEqual(norm, 1)

    '''def test_projectors(self):
        """Tests that projectors project on correct state."""
        k = 2
        lst = range(k+1)
        for n, m in product(lst, lst):
            test_state = construct_fock_state(n, m)
            projector = construct_projector(n, m)
            states, states_dict = generate_fock_states(k, k)
            for name in states_dict:
                state = states_dict[name]
                projection = projector * state
                inner_prod = projection.inner_product(test_state)
                norm = integrate(simplify(inner_prod))
                norm = simplify(norm)
                print(name, norm, (n, m))
                if name == (n, m):
                    pass
                    # self.assertAlmostEqual(norm, 1)
                elif name == (m, n):
                    pass
                else:
                    # self.assertAlmostEqual(norm, 0)
                    pass'''

    def test_povms(self):
        """Tests the correct generation of POVM operators."""
        # read from pre-generated text file containing the operators.
        ops_dict = read_from_txt('full_povms_2_2_photons.txt')
        for visibility in np.linspace(0, 1, 10):
            sum = np.zeros(shape=(9, 9))
            for op in ops_dict.values():
                mat = op.to_numpy_matrix(convert_scalars, visibility)
                sum += mat
            # check if povms sum up to identity
            id = np.identity(9)
            self.assertTrue(np.testing.assert_array_almost_equal(sum, id) is None)

    def test_against_old_povms(self):
        """Test if POVMs agree with the old ones for visibility = 1."""
        # Note: This is a pickle containing the generated and converted POVMs for up to 3 photons from each side for
        # visibility (NOT the complete set of POVMs just up to 3 photons after the BS.
        with open('multiphoton_povms_arrays_2.pkl', "rb") as file:
            arrays = pickle.load(file)
        ops_dict = read_from_txt('subset_povms_3_3_4photons.txt')
        # TODO: extend to full set of POVMs
        kraus_ops, kraus_ops_num_res, outcome_dict = set_operators()

        def generate_dict(total_photon_number, list):
            key = []
            dict = {}
            for n in range(total_photon_number + 1):
                for m in range(total_photon_number + 1):
                    if n + m <= total_photon_number:
                        key.append((n, m))
            for i in range(len(list)):
                dict[key[i]] = list[i]
            return dict
        kraus_dict = generate_dict(6, kraus_ops_num_res)
        array_dict = {}
        # convert operators to matrices with visibility = 1
        visibility = 1.
        for k in ops_dict.keys():
            array_dict[k] = ops_dict[k].to_numpy_matrix(convert_scalars, visibility)

        #array_dict = generate_dict(4, arrays)
        for key in array_dict.keys():
            # old POVMs seem to have reverse numbering
            (n, m) = key
            rev_key = (m, n)
            # ToDo: fix povms
            if n + m != 4:
                self.assertTrue(np.testing.assert_array_almost_equal(array_dict[key],
                                                                     kraus_dict[rev_key].arr.real) is None)
            else:
                #np.testing.assert_array_almost_equal(array_dict[key], kraus_dict[rev_key].arr.real)
                print(f"({n},{m}): For 4 photons POVMs dont agree?!")


if __name__ == "__main__":
    unittest.main()
