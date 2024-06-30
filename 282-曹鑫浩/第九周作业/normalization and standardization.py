import numpy as np


def normalization_1(lst):
    return [(lst[i]- min(lst))/ (max(lst)- min(lst)) for i in range(len(lst))]


def normalization_2(lst):
    return [(lst[i]- np.mean(lst))/ (max(lst)- min(lst)) for i in range(len(lst))]


def standardization(lst):
    e = lambda lst1: sum([(lst1[i] - np.mean(lst1)) ** 2 for i in range(len(lst1))]) / len(lst1)
    return[(lst[i] - np.mean(lst))/ e(lst) for i in range(len(lst))]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

a = normalization_1(l)
b = normalization_2(l)
c = standardization(l)
print(a)
print(b)
print(c)