import unittest
from dtl_algebra import (
    generate_dtl_states,
    apply_identity,
    apply_half_vacuum_1,
    apply_half_vacuum_2,
    apply_half_vacuum_3,
    apply_half_vacuum_4,
    apply_tl,
)

class TestDTLAlgebra(unittest.TestCase):
    def test_generate_dtl_states(self):
        # L=3, j=1
        states_3_1 = generate_dtl_states(3, 1)
        expected_3_1 = [(1, 1, -1), (1, 0, 0), (1, -1, 1), (0, 1, 0), (0, 0, 1)]
        self.assertEqual(set(states_3_1), set(expected_3_1))

        # L=2, j=0
        states_2_0 = generate_dtl_states(2, 0)
        expected_2_0 = [(1, -1), (0, 0)]
        self.assertEqual(set(states_2_0), set(expected_2_0))

    def test_identity(self):
        self.assertEqual(apply_identity((1, -1), 0), [((1, -1), 1)])
        self.assertEqual(apply_identity((1, 1), 0), [((1, 1), 1)])
        self.assertEqual(apply_identity((-1, 1), 0), [((-1, 1), 1)])
        self.assertEqual(apply_identity((-1, -1), 0), [((-1, -1), 1)])
        self.assertEqual(apply_identity((1, 0), 0), [])
        self.assertEqual(apply_identity((0, 0), 0), [])

    def test_half_vacuum_1(self):
        self.assertEqual(apply_half_vacuum_1((1, 0), 0), [((1, 0), 1)])
        self.assertEqual(apply_half_vacuum_1((-1, 0), 0), [((-1, 0), 1)])
        self.assertEqual(apply_half_vacuum_1((0, 1), 0), [])
        self.assertEqual(apply_half_vacuum_1((1, 1), 0), [])

    def test_half_vacuum_2(self):
        self.assertEqual(apply_half_vacuum_2((0, 1), 0), [((0, 1), 1)])
        self.assertEqual(apply_half_vacuum_2((0, -1), 0), [((0, -1), 1)])
        self.assertEqual(apply_half_vacuum_2((1, 0), 0), [])
        self.assertEqual(apply_half_vacuum_2((1, 1), 0), [])

    def test_half_vacuum_3(self):
        self.assertEqual(apply_half_vacuum_3((1, 0), 0), [((0, 1), 1)])
        self.assertEqual(apply_half_vacuum_3((-1, 0), 0), [((0, -1), 1)])
        self.assertEqual(apply_half_vacuum_3((0, 1), 0), [])
        self.assertEqual(apply_half_vacuum_3((1, 1), 0), [])

    def test_half_vacuum_4(self):
        self.assertEqual(apply_half_vacuum_4((0, 1), 0), [((1, 0), 1)])
        self.assertEqual(apply_half_vacuum_4((0, -1), 0), [((-1, 0), 1)])
        self.assertEqual(apply_half_vacuum_4((1, 0), 0), [])
        self.assertEqual(apply_half_vacuum_4((1, 1), 0), [])

    def test_tl_generator(self):
        n = 2.5
        # (1, -1) -> (1, -1) with weight n
        self.assertEqual(apply_tl((1, -1), 0, n), [((1, -1), n)])

        # (-1, 1) -> (1, -1) with weight 1
        self.assertEqual(apply_tl((-1, 1), 0, n), [((1, -1), 1)])

        # touches 0 -> []
        self.assertEqual(apply_tl((1, 0), 0, n), [])
        self.assertEqual(apply_tl((0, 1), 0, n), [])

        # (1, 1) matching
        state1 = (1, 1, 1, -1, -1)
        res1 = apply_tl(state1, 0, n) # acts on (1, 1) at i=0
        self.assertEqual(res1, [((1, -1, 1, -1, 1), 1)])

        # (-1, -1) matching
        state2 = (1, -1, 1, -1, -1)
        res2 = apply_tl(state2, 3, n) # acts on (-1, -1) at i=3
        self.assertEqual(res2, [((1, -1, -1, 1, -1), 1)])

        # No match case
        self.assertEqual(apply_tl((1, 1), 0, n), [])
        self.assertEqual(apply_tl((-1, -1), 0, n), [])

    def test_constraints_preserved(self):
        # We verify that any non-zero application of TL keeps state in the same module
        # Module V^{L,j} has sum = j, partial sums >= 0.
        L = 5
        j = 1
        states = generate_dtl_states(L, j)
        for state in states:
            for i in range(L - 1):
                results = apply_tl(state, i, 2.0)
                for res_state, weight in results:
                    # Check partial sums >= 0
                    partial_sum = 0
                    for s in res_state:
                        partial_sum += s
                        self.assertGreaterEqual(partial_sum, 0)
                    # Check total sum == j
                    self.assertEqual(partial_sum, j)

if __name__ == "__main__":
    unittest.main()
