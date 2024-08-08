
import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])
condition = a > 3  # 假设条件是a中的元素大于3
result = np.where(condition, a, b)  # 根据条件从a和b中选择元素
print(result)  # 输出满足条件的a中的元素