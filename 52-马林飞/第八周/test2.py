import numpy as np
from scipy.linalg import lstsq

# 系数矩阵 A 和右手边向量 b
A = np.array([[1, 1], [1, 2], [1, 3]])
b = np.array([1, 2, 2])

# 使用最小二乘法求解 Ax = b
x, residues, rank, s = lstsq(A, b)

print("Solution x:")
print(x)
print("Residues:")
print(residues)
print("Rank of A:")
print(rank)
print("Singular values of A:")
print(s)


print(7e3)