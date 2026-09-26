import unittest
from assignment1 import Vec


class TestMean(unittest.TestCase):

    # Normal vector
    def test_mean(self):
        v = Vec([1, 5, 6])
        self.assertEqual(v.mean(), 4)

    # Vector containing negative numbers
    def test_mean_with_negative_numbers(self):
        v = Vec([-2, 4, 6])
        self.assertAlmostEqual(v.mean(), 8 / 3)

    # Vector with decimal values
    def test_mean_with_decimal_values(self):
        v = Vec([1.5, 2.5, 3.5])
        self.assertEqual(v.mean(), 2.5)

    # Mean of a constant vector
    def test_mean_constant_vector(self):
        v = Vec([5, 5, 5, 5])
        self.assertEqual(v.mean(), 5)

    # Mean of a single-element vector
    def test_mean_single_element(self):
        v = Vec([10])
        self.assertEqual(v.mean(), 10)

    # Empty vector should raise an error
    def test_mean_empty_vector(self):
        v = Vec([])

        with self.assertRaises(ValueError):
            v.mean()

class TestDemean(unittest.TestCase):
    #testing the actual demeaned vector
    def test_demean(self):
        v = Vec([1,5,6])
        result = v.demean()
        self.assertEqual(result.elements, [-3,1,2])

    #mean of a demeaned vector should be 0
    def test_demean_mean(self):
        v = Vec([1,5,6])
        result = v.demean()
        self.assertEqual(result.mean(), 0)

if __name__ == "__main__":
    unittest.main()