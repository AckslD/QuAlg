import unittest
import time

from multiphoton_povms import generate_fock_states
from integrate import integrate
from toolbox import simplify
import scalars


class TestMultiPhoton(unittest.TestCase):
    """Test the correct implementation of multiphoton states."""

    def test_state_generation(self):
        """Test the correct geneeration of Fock states."""
        n = 3
        states = generate_fock_states(n, n)
        # states2 = generate_fock_states(n, m)

        for state in states:
            print("length of state:", len(state))
            start_time = time.time()
            #print("State:", state)
            inner_prod = state.inner_product(state)
            ip = simplify(inner_prod)
            #print("Inner Prod", ip)
            #print("repr", repr(ip))
            #if ip is not isinstance(scalars.ComplexScalars):
            #    for f in ip._factors:
            #        print(type(f))
            norm = integrate(ip)
            print("Norm:", simplify(norm))
            #print(repr(simplify(norm)))
            elapsed_time = time.time() - start_time
            print("took {}s to calculate".format(elapsed_time))
            #self.assertAlmostEqual(norm._key(), 1)


if __name__ == "__main__":
    unittest.main()
