import unittest
import time


from multiphoton_povms import generate_fock_states
from integrate import integrate




class TestMultiPhoton(unittest.TestCase):
    """Test the correct implementation of multiphoton states."""

    def test_state_generation(self):
        """Test the correct geneeration of Fock states."""
        n = 3
        m = 3
        states = generate_fock_states(n, m)
        # states2 = generate_fock_states(n, m)
        print(len(states))
        for state in states:
            start_time = time.time()
            print("State:", state)
            inner_prod = state.inner_product(state)
            norm = integrate(inner_prod)
            print("Norm:", norm)
            elapsed_time = time.time() - start_time
            print("took {}s to calculate".format(elapsed_time))
            #self.assertAlmostEqual(norm._key(), 1)



if __name__ == "__main__":
    unittest.main()