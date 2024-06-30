import numpy as np

a = np.array([1, 2, 3])
print(a)

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)

c = np.zeros((3, 4))
print(c)

d = np.ones((2, 2))
print(d)

# 创建一个全等数组，每个元素的值都为某个给定值
e = np.full((3, 3), 7)
print(e)

f = np.arange(5)
print(f)

print(a.shape)
print(a.ndim)
print(a.dtype)

print(a[0])
print(a[:3])
print(b[0, 0])
print(b[0, :])

a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
result = a + b
print(a + b)


print(np.square(a))
print(np.sqrt(a))
print(np.exp(a))

print(np.sum(result))
print(np.mean(result))
print(np.max(result))
print(np.std(result))

