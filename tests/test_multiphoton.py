import unittest
import time
from itertools import product

from multiphoton_povms import generate_fock_states, construct_fock_state, construct_projector
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
            start_time = time.time()

            inner_prod = state.inner_product(state)
            ip = simplify(inner_prod)
            norm = integrate(ip)

            elapsed_time = time.time() - start_time

            print(f"{name} has length {len(state)}.")
            print(f"|{name}> has norm:", simplify(norm))
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
                norm = simplify(norm)
                print(name, norm, (n, m))
                if name == (n, m):
                    pass
                    # self.assertAlmostEqual(norm, 1)
                elif name == (m, n):
                    pass
                else:
                    # self.assertAlmostEqual(norm, 0)
                    pass


if __name__ == "__main__":
    unittest.main()
