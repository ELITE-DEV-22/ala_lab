import numpy as np
import unittest

#Q.1 a) Construct D programmatically (no hard-coded matrix literal) for a general size n (n-1), then compute xTD

def implement_D(n):
    D = np.zeros((n, n-1), dtype = np.float64)#creates a matrix filled with zeros for n and n-1 dim
    #logic to add the pattern of -1 and 1
    for j in range (n -1):
        D[j, j] = -1.0
        D[j + 1, j] = 1.0
    return D

x = np.array([1,4,9,16,25,36], dtype = np.float64)#create the second array to multiply it with
D = implement_D(len(x))
result = x @ D #matrix multiplication 
print(D)
print("\n x^T D =", result)
print("\n Shape of result:", result.shape)

# (b) Write a function that returns the subvector x2:5 using the convention above and use it to obtain x2∶5.
def slice_vec(x,i,j):
    return x[i - 1:j]#return the slice using 1 based inclusive indexing by converting it to NumPy 0-based slicing
slice = slice_vec(x,2,5)
print("\n x(2:5)", slice)

# (c) Compute d = x2∶6 − x1∶5 and verify numerically that d = DTx (equivalently,xTD from part (a)).
x_2_6 = slice_vec(x, 2, 6)
x_1_5 = slice_vec(x, 1, 5)

d = x_2_6 - x_1_5

d_matrix = D.T @ x
print("\n x_2:6 =", x_2_6)
print("\n x_1:5 =", x_1_5)
print("\n d =", d)
print("\n D^T x =", d_matrix)
print("\n Verification:", np.allclose(d, d_matrix))#checks if both results match 

# d) unit test for a,b,c
class TestQuestion1(unittest.TestCase):

    def setUp(self):
        #initialize the vector and construct D before each test
        self.x = np.array([1, 4, 9, 16, 25, 36], dtype=np.float64)
        self.D = implement_D(len(self.x))

    def test_implement_D(self):
        #test if D has the correct shape and column pattern
        self.assertEqual(self.D.shape, (6, 5))

        for j in range(self.D.shape[1]):
            expected_column = np.zeros(6, dtype=np.float64)
            expected_column[j] = -1.0
            expected_column[j + 1] = 1.0
            self.assertTrue(np.allclose(self.D[:, j], expected_column))

    def test_slice_vec(self):
        #test if slice_vec returns the correct subvector x_2:5
        expected = np.array([4, 9, 16, 25], dtype=np.float64)
        actual = slice_vec(self.x, 2, 5)
        self.assertTrue(np.allclose(actual, expected))

    def test_difference(self):
        #test if the subvector difference matches D^T x
        d = slice_vec(self.x, 2, 6) - slice_vec(self.x, 1, 5)
        d_matrix = self.D.T @ self.x

        expected = np.array([3, 5, 7, 9, 11], dtype=np.float64)

        self.assertTrue(np.allclose(d, expected))
        self.assertTrue(np.allclose(d, d_matrix))

# QUESTION 2
class question2:
    def __init__(self, D):
        self.D = D
    def compute(self):
        X = np.random.default_rng(0).standard_normal((6, 768))#generate six token embeddings each containing 768 values
        result  = self.D.T @ X #create an obejct and compute D.T @ X
        return X, result
q2 = question2(D)
X, result_X = q2.compute()
print("\n Shape of X:", X.shape)
print("\n Shape of D^T X:", result_X.shape)

assert X.shape == (6, 768)#assert X has the correct shape
assert result_X.shape == (5, 768)#assert result has correct shape


#QUESTION 3

class question3:
    def layer_norm(self, e, gamma, beta, eps):
        m = np.sum(e)/ len (e)#calculate the scalar mean 
        v = np.sum((e-m)**2) / len(e)#calcuate the variance
        normalized = (e-m) / np.sqrt(v+eps)#normalize the input vector
        output = normalized * gamma + beta
        return output
q3 = question3()

#generate input activation vector of size 768
rng = np.random.default_rng(0)
e = rng.standard_normal(768).astype(np.float64)
#initialize the gamma and beta values
gamma = np.ones(768, dtype=np.float64)
beta = np.zeros(768, dtype=np.float64)
#set the value of epilison
eps = 1e-5
#apply the layer normalization
output = q3.layer_norm(e, gamma, beta, eps)
#display the input and output shape
print("\nShape of input:", e.shape)
print("\nShape of output:", output.shape)

# Unit test for Question 3
# Unit tests for Question 3

class TestQuestion3(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.e = rng.standard_normal(768).astype(np.float64)
        self.gamma = np.ones(768, dtype=np.float64)
        self.beta = np.zeros(768, dtype=np.float64)
        self.eps = 1e-5
        self.q3 = question3()

    def test_reference_formula(self):
        #compare implementation with the reference formula
        m = np.mean(self.e)
        v = np.var(self.e)

        expected = ((self.e - m) / np.sqrt(v + self.eps)
                    ) * self.gamma + self.beta

        actual = self.q3.layer_norm(
            self.e, self.gamma, self.beta, self.eps
        )

        self.assertTrue(
            np.allclose(actual, expected, rtol=1e-10, atol=1e-12)
        )

    def test_output_mean_and_variance(self):
        #verify output mean and variance when gamma = 1 and beta = 0
        output = self.q3.layer_norm(
            self.e, self.gamma, self.beta, self.eps
        )

        v = np.var(self.e)
        expected_variance = v / (v + self.eps)

        self.assertAlmostEqual(
            float(np.mean(output)), 0.0, delta=1e-10
        )
        self.assertAlmostEqual(
            float(np.var(output)),
            float(expected_variance),
            delta=1e-10
        )

if __name__ == "__main__":
    unittest.main()