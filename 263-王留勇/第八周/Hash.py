"""
Hash
"""

import cv2
import random
import matplotlib.pyplot as plt
import numpy as np


# 图像加噪
def addNoise(img):
	rows, cols, channels = img.shape
	noiseNum = random.randint(1, rows * cols)  # 随机噪声数
	for i in range(noiseNum):
		# 随机位置
		row = random.randint(1, rows - 1)
		col = random.randint(1, cols - 1)

		for j in range(channels):
			colorValue = random.randint(0, 255)
			img[row, col, j] = colorValue
	return img


# 均值哈希算法
def aHash(img):
	img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	sum = 0  # 像素综合
	hash_str = ''  # 初始hash值
	for i in range(8):
		for j in range(8):
			sum += gray[i, j]
	average = sum / 64  # 像素均值

	# 灰度大于平均值为1否则为0
	for i in range(8):
		for j in range(8):
			if gray[i, j] > average:
				hash_str = hash_str + '1'
			else:
				hash_str = hash_str + '0'
	return hash_str


# 差值哈希算法
def dHash(img):
	img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	hash_str = ''  # 初始hash值
	for i in range(8):
		for j in range(8):
			if gray[i, j] > gray[i, j + 1]:
				hash_str = hash_str + '1'
			else:
				hash_str = hash_str + '0'
	return hash_str

# 感知哈希算法
def pHash(img, width=64, hight=64):
	img = cv2.resize(img, (width, hight), interpolation=cv2.INTER_CUBIC)

	rows, cols = img.shape[:2]
	vis0 = np.zeros((rows, cols), np.float32)
	vis0[:rows, :cols] = img

	# 二维Dct变换
	vis1 = cv2.dct(cv2.dct(vis0))
	vis1.resize(32, 32)

	img_list = vis1.flatten()

	# 计算均值
	avg = sum(img_list) * 1. / len(img_list)
	avg_list = ['0' if i > avg else '1' for i in img_list]
	hash_str = ''.join(['%x' % int(''.join(avg_list[x:x+4]), 2) for x in range(0, 32 * 32, 4)])
	return hash_str

# Hash 值对比
def cmpHash(hash1, hash2):
	if len(hash1) != len(hash2):
		return -1
	n = 0  # 记录不同位数
	# 遍历判断
	for i in range(len(hash1)):
		if hash1[i] != hash2[i]:
			n += 1
	return n


img = cv2.imread('lenna.png')
img_noise = addNoise(img.copy())

hash1 = aHash(img)
hash2 = aHash(img_noise)
print('aHash-hash1=', hash1)
print('aHash-hash2=', hash2)
n = cmpHash(hash1, hash2)
print('均值哈希差异值：', n)

hash1 = dHash(img)
hash2 = dHash(img_noise)
print('dHash-hash1=', hash1)
print('dHash-hash2=', hash2)
n = cmpHash(hash1, hash2)
print('差值哈希差异值：', n)

hash1 = pHash(img)
hash2 = pHash(img_noise)
print('pHash-hash1=', hash1)
print('pHash-hash2=', hash2)
n = cmpHash(hash1, hash2)
print('感知哈希差异值：', n)

cv2.imshow('src img', img)
cv2.imshow('noise img', img_noise)
cv2.waitKey(0)
cv2.destroyWindow()





















