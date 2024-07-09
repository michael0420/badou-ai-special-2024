# -*- coding: utf-8 -*-
"""
@File    :   标准.py
@Time    :   2024/06/21 13:47:04
@Author  :   廖红洋 
"""
import numpy as np
import matplotlib.pyplot as plt

input = [
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    5,
    5,
    6,
    6,
    7,
    8,
]
output = []

# 生成数据统计图
fn = []
s = 0.0
for i in input:
    s += i
    c = input.count(i)
    fn.append(c)
s /= len(input)
l = max(input) - min(input)

for i in input:
    tmp = ((float)(i) - s) / l
    output.append(tmp)

idx = np.arange(len(input))
plt.plot(input, fn, "-", label="原始数据")
plt.plot(output, fn, "-", label="标准化数据")
plt.rcParams["font.sans-serif"] = ["simHei"]
plt.legend()
plt.show()
