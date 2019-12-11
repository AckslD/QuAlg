import unittest
import time
from itertools import product
import numpy as np
import pickle

from netsquid_ae.qdetector_multi import set_operators

from multiphoton_povms import generate_fock_states, construct_fock_state, construct_projector, generate_effective_povms,\
    convert_scalars
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
            print(f"{name} has length {len(state)}.")
            start_time = time.time()

            inner_prod = state.inner_product(state)
            print("inner took:", time.time() - start_time)
            ip = simplify(inner_prod)
            ip = simplify(ip)

            norm = integrate(ip)

            elapsed_time = time.time() - start_time

            print(f"|{name}> has norm:", norm, type(norm))
            print(f"Norm took {elapsed_time}s to calculate")

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
        # n = 2
        # povm_ops = generate_effective_povms(n, n)

        # Note: this is a pickle containing the generated and converted povm arrays for up to 2 photons from each side
        # with visibility = 1.
        # generating the operators took:    456.10833168029785 s
        # converting to arrays took:        2519.8453447818756 s  (factor 5.5 longer)
        with open('multiphoton_povms_arrays_2_2_2.pkl', "rb") as file:
            arrays = pickle.load(file)
        sum = np.zeros(shape=(9, 9))
        for mat in arrays:
            sum += mat

        id = np.identity(9)
        self.assertTrue(np.testing.assert_array_almost_equal(sum, id) is None)

    def test_against_old_povms(self):
        """Test if POVMs agree with the old ones for visibility = 1."""
        # Note: This is a pickle containing the generated and converted POVMs for up to 3 photons from each side for
        # visibility (NOT the complete set of POVMs just up to 3 photons after the BS.
        with open('multiphoton_povms_arrays_2.pkl', "rb") as file:
            arrays = pickle.load(file)
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

        array_dict = generate_dict(3, arrays)
        kraus_dict = generate_dict(6, kraus_ops_num_res)

        for key in array_dict.keys():
            # old POVMs seem to have reverse numbering
            (n, m) = key
            rev_key = (m, n)
            self.assertTrue(np.testing.assert_array_almost_equal(array_dict[key], kraus_dict[rev_key].arr.real) is None)


if __name__ == "__main__":
    unittest.main()
