from numba import jit
from numpy import arange
from time import time

# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
@jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result


n = 10000
a = arange(n**2).reshape(n,n)
t = time()

nrun = 10
for _ in range(nrun):
	print(sum2d(a))
	
print("time =", (time()-t)/nrun)
