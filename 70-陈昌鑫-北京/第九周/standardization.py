import numpy as np
import matplotlib.pyplot as plt

def Normalization_1(list):
    '''
    (0~1)
    '''
    return [(float(i) - min(list)) / float(max(list) - min(list)) for i in list]

def Normalization_2(list):
    '''
    (-1~1)
    '''
    return [(float(i) - np.mean(list)) / (max(list) - min(list)) for i in list]

def z_score(list):
    '''
    x∗=(x−μ)/σ
    '''
    x_mean = np.mean(list)
    sum_list = sum([(i - np.mean(list)) * (i - np.mean(list)) for i in list]) / len(list)
    return [(i - x_mean) / sum_list for i in list]


list = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs=[]
for i in list:
    num=list.count(i)
    cs.append(num)
print(cs)

norm = Normalization_1(list)
z_score = z_score(list)
plt.plot(list, cs)
plt.plot(z_score, cs)
plt.show()