import unittest
import time
from itertools import product
import numpy as np
from scipy.linalg import sqrtm

from netsquid_ae.qdetector_multi import set_old_operators

from QuAlg.multiphoton_povms import generate_fock_states, construct_projector, generate_effective_povms, convert_scalars
from QuAlg.multiphoton_tools import read_from_txt
from QuAlg.integrate import integrate
from QuAlg.toolbox import simplify, get_variables


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

    def test_projectors(self):
        """Tests that projectors project on correct state."""
        pnum = 2
        lst = range(pnum+1)
        states, states_dict = generate_fock_states(pnum, pnum)
        for name in states_dict:
            for n, m in product(lst, lst):
                start_time = time.time()
                projector = construct_projector(n, m)
                state = states_dict[name]
                projection = projector * state
                # simplify
                if not isinstance(projection, int):
                    for base_op, scalar in projection._terms.items():
                        scalar_variables = get_variables(scalar) - get_variables(base_op)
                        projection._terms[base_op] = integrate(scalar, scalar_variables)
                inner_prod = projection.inner_product(state)
                norm = integrate(simplify(inner_prod))
                print(f"<{name}|P{n, m}|{name}> = {norm}")
                if time.time()-start_time > 1:
                    print(f"calculation took: {time.time()-start_time}s")
                (k, l) = name
                if k + l != n + m:
                    self.assertAlmostEqual(norm, 0)
                else:
                    self.assertNotEqual(norm, 0)

    def test_povms(self):
        """Tests the correct generation of POVM operators."""
        # read from pre-generated text file containing the operators.
        ops_dict = read_from_txt('QuAlg/full_multiphoton_povms.txt')
        for visibility in np.linspace(0, 1, 10):
            sum_povms = np.zeros(shape=(16, 16))
            for op in ops_dict.values():
                mat = op.to_numpy_matrix(convert_scalars, visibility)
                sum_povms += mat
            # check if povms sum up to identity
            idnt = np.identity(16)
            self.assertTrue(np.testing.assert_array_almost_equal(sum_povms, idnt) is None)

    def test_against_ll(self):
        """Tests if the generation script can reproduce the link layer povms."""
        # generate link layer arrays
        def set_ll_operators_num_resolving(visibility):
            """Sets up the relevant operators used for the measurement, depending on the state formalism used.

            Note: These are the operators for a number resolving detector.

            """
            # Assuming mu is real
            mu = np.sqrt(visibility)
            s_plus = (np.sqrt(1 + mu) + np.sqrt(1 - mu)) / (2. * np.sqrt(2))
            s_min = (np.sqrt(1 + mu) - np.sqrt(1 - mu)) / (2. * np.sqrt(2))
            # the Kraus operator for measuring zero photons
            M_00 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            # the Kraus operator for measuring one photon from station A
            M_10 = \
                np.array([[0, 0, 0, 0],
                          [0, s_plus, s_min, 0],
                          [0, s_min, s_plus, 0],
                          [0, 0, 0, 0]])
            # the Kraus operator for measuring one photon from station B
            M_01 = \
                np.array([[0, 0, 0, 0],
                          [0, s_plus, -1. * s_min, 0],
                          [0, -1. * s_min, s_plus, 0],
                          [0, 0, 0, 0]])
            # the Kraus operator for measuring two photons, one on each side
            M_11 = \
                np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, np.sqrt(1 - mu * mu) / np.sqrt(2)]])
            # the Kraus operator for measuring two photons from station A
            M_20 = \
                np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, np.sqrt(1 + mu * mu) / 2]])
            # the Kraus operator for measuring two photons from station B
            M_02 = M_20

            dict = {(0, 0): M_00,
                    (0, 1): M_01,
                    (0, 2): M_02,
                    (1, 0): M_10,
                    (1, 1): M_11,
                    (2, 0): M_20}

            return dict

        # generate povms
        ops_dict = generate_effective_povms(1, 1)
        #for visibility in [0, 0.1, 1]:
        for visibility in np.linspace(0, 1, 10):
            # generate ll povms with current visibility
            ll_ops_dict = set_ll_operators_num_resolving(visibility)
            # translate new povms to arrays with current visibility
            array_dict = {}
            for k in ops_dict.keys():
                array_dict[k] = sqrtm(ops_dict[k].to_numpy_matrix(convert_scalars, visibility))
            # compare arrays
            for key in array_dict.keys():
                self.assertTrue(np.testing.assert_array_almost_equal(array_dict[key], ll_ops_dict[key]) is None)

    def test_against_old_povms(self):
        """Test if POVMs agree with the old ones for visibility = 1."""
        # import new povms from txt file
        ops_dict = read_from_txt('QuAlg/full_multiphoton_povms.txt')
        # import old povms from qdetector
        kraus_ops, kraus_ops_num_res, outcome_dict = set_old_operators()

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
            array_dict[k] = sqrtm(ops_dict[k].to_numpy_matrix(convert_scalars, visibility))

        #array_dict = generate_dict(4, arrays)
        for key in array_dict.keys():
            # old POVMs seem to have reverse numbering
            (n, m) = key
            rev_key = (m, n)
            # ToDo: fix povms
            #print(key, np.testing.assert_array_almost_equal(array_dict[key], kraus_dict[rev_key].arr.real))
            print(key, np.allclose(array_dict[key], kraus_dict[rev_key].arr.real))

            #if not np.allclose(array_dict[key], kraus_dict[rev_key].arr.real):
            #    print(array_dict[key])
            #    print(kraus_dict[rev_key].arr.real)

            #self.assertTrue(np.testing.assert_array_almost_equal(array_dict[key],
            #                                                     kraus_dict[rev_key].arr.real) is None)


if __name__ == "__main__":
    unittest.main()
