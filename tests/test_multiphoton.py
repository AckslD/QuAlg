import unittest
import time
from itertools import product

from multiphoton_povms import generate_fock_states, construct_fock_state, construct_projector
from integrate import integrate
from toolbox import simplify


class TestMultiPhoton(unittest.TestCase):
    """Test the correct implementation of multiphoton states."""

    def test_state_generation(self):
        """Test the correct geneeration of Fock states."""
        n = 2
        states, states_dict = generate_fock_states(n, n)

        for name in states_dict:
            state = states_dict[name]
            print(f"{name} has length {len(state)}.")
            start_time = time.time()
            inner_prod = state.inner_product(state)
            ip = simplify(inner_prod)
            norm = integrate(ip)
            print(f"|{name}> has norm:", simplify(norm))
            elapsed_time = time.time() - start_time
            print(f"Norm took {elapsed_time}s to calculate")

            # currently only works for phi_i = phi_j m psi_i = psi_j
            self.assertAlmostEqual(norm, 1)

    def test_projectors(self):
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
                print(name, norm, (n, m))
                if name == (n, m):
                    self.assertAlmostEqual(norm, 1)
                else:
                    self.assertAlmostEqual(norm, 0)


if __name__ == "__main__":
    unittest.main()
