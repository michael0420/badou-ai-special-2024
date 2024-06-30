'''
第九周作业
1.实现参数标准化
2.手推训练过程
'''

import matplotlib.pyplot as plt
import numpy as np
# 1.实现参数标准化
# 创建数据归一化方法，将数据统一映射到[0,1]之间
# 数学公式y=(x-min)/(max-min)
def NormaLization(list):
    listnor=[]
    for i in list:
        sum=(float(i)-min(list))/float(max(list)-min(list))
        listnor.append(sum)
    return listnor

# 创建z-score标准化方法（零均值归一化zero-mean normalization）：
# 数学公式y=(x-μ)/σ 其中μ是样本的均值， σ是样本的标准差

def ZeroMean(list):
    listzero=[]
    listmean=np.mean(list)
    sum=0
    for i in list:
        sum+=(i-listmean)*(i-listmean)
    sum2=sum/len(list)
    for i in list:
        s=(i-listmean)/sum2
        listzero.append(s)
    return listzero

list=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
listnorma=NormaLization(list)
print(listnorma)

listzero=ZeroMean(list)
print(listzero)

cs=[]
for i in list:
    c=list.count(i)
    cs.append(c)

'''
蓝线为原始数据，橙线为z
'''
plt.plot(list,cs)
plt.plot(listzero,cs)
plt.show()




