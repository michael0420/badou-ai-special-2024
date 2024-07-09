import math

import numpy as np


def ReasoningTraining(i1, i2):
	w1 = 0.15
	w2 = 0.20
	w3 = 0.25
	w4 = 0.30

	w5 = 0.40
	w6 = 0.45
	w7 = 0.50
	w8 = 0.55

	b1 = 0.35
	b2 = 0.60

	o1 = 0.01
	o2 = 0.99

	x = 0.5  # 学习率

	zh1 = w1 * i1 + w2 * i2 + b1 * 1
	print('zh1=', zh1)
	zh2 = w3 * i1 + w4 * i2 + b1 * 1
	print('zh2=', zh2)

	ah1 = 1 / (1 + math.exp(-zh1))
	ah2 = 1 / (1 + math.exp(-zh2))
	print('ah1=', ah1)
	print(('ah2=', ah2))

	zo1 = w5 * ah1 + w6 * ah2 + b2 * 1
	zo2 = w7 * ah1 + w8 * ah2 + b2 * 1

	ao1 = 1 / (1 + math.exp(-zo1))
	ao2 = 1 / (1 + math.exp(-zo2))
	print('ao1=', ao1)
	print('ao2=', ao2)

	Eo1 = ((o1 - ao1)**2) / 2
	Eo2 = ((o2 - ao2)**2) / 2
	Etotal = Eo1 + Eo2
	print('Eo1=', Eo1, '\nEo2=', Eo2, '\nEtotal=', Etotal)

	Et_ao1 = 2 * 1/2 * (o1 - ao1) * (-1)
	Et_ao2 = 2 * 1/2 * (o2 - ao2) * (-1)

	ao1_zo1 = ao1 * (1 - ao1)
	ao2_zo2 = ao2 * (1 - ao2)

	zo1_w5 = ah1
	zo2_w6 = ah2

	Etotal_w5 = Et_ao1 * ao1_zo1 * zo1_w5

	p1 = - (o1 - ao1) * ao1 * (1 - ao1)
	p1_w5 = p1 * ah1

'''
同理，求出 w1, w2, w3, w4, w6, w7, w8 的导数
使用公式： 新权重 = 旧权重 - 学习率 * 对应导数
重复 25行-60行代码， 得出结果与 o1, o2 比较，直到小于目标误差再终止
w1-w8即为训练过程每一步的权重值
'''






ReasoningTraining(0.05, .10,)
