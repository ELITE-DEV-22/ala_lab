import numpy as np

#Q.1 a) Construct D programmatically (no hard-coded matrix literal) for a general size n (n-1), then compute xTD

def implement_D(n):
    D = np.zeros((n, n-1), dtype = np.float64)#creates a matrix filled with zeros for n and n-1 dim
    #logic to add the pattern of -1 and 1
    for j in range (n -1):
        D[j, j] = -1.0
        D[j + 1, j] = 1.0
    return D

x = np.array([1,4,9,16,25,36], dtype = float)#create the second array to multiply it with
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
print("\nx_2:6 =", x_2_6)
print("\nx_1:5 =", x_1_5)
print("\nd =", d)
print("\nD^T x =", d_matrix)
print("\nVerification:", np.allclose(d, d_matrix))#checks if both results match w