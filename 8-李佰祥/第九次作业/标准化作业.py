import numpy as np
l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]



#0-1归一化  'x_=(x−x_min)/(x_max−x_min)
result = [(float(i)-min(l))/float(max(l)-min(l)) for i in l]

#print(result)

#-1到1归一化： x = (x - x_mean)/(x_max-x_min)
result2  = [(float(i) - np.mean(l))/(max(l) - min(l)) for i in l]
#print(result2)


#零均值归一化:均值为0，标准差为1
#y = (x - u)/o,u为样本均值，o为样本的标准差
u = np.mean(l)
o = sum([(i-np.mean(l) * i-np.mean(l)) for i in l]) / len(l)
result3 = [(i - u)/o for i in l]
print(result3)


