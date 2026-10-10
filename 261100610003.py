import numpy as np

#Q.1 a) Construct D programmatically (no hard-coded matrix literal) for a general size n (n-1), then compute xTD

def implement_D(n):
    D = np.zeros((n, n-1), dtype = np.float64)

    for j in range (n -1):
        D[j, j] = -1.0
        D[j + 1, j] = 1.0
    return D

x = np.array([1,4,9,16,25,36], dtype = float)
D = implement_D(len(x))
result = x @ D
print(D)

print("x^T D =", result)
print("Shape of result:", result.shape)
