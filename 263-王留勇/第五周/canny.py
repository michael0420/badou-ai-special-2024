"""
canny 边缘检测
"""

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

# 图片读取
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gaussianImg = cv2.GaussianBlur(gray, (3, 3), 0)

# 普通 canny 边缘检测
def cannyImg():
	canny_img = cv2.Canny(gray, 100, 200)
	cv2.imshow('cource')
	cv2.waitKey(0)

# 可调节 canny 边缘检测
lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

# 调解杠变化函数
def cannyThreshold(lowThreshold):
	detected_edges = cv2.Canny(gaussianImg, lowThreshold, lowThreshold*ratio, apertureSize=kernel_size)
	dst = cv2.bitwise_and(img, img, mask=detected_edges)
	cv2.imshow('canny result', dst)

# 设置调解杠
def canny_track():
	cv2.namedWindow('canny result')
	cv2.createTrackbar('Min threshold', 'canny result', lowThreshold, max_lowThreshold, cannyThreshold)
	cannyThreshold(0)


"""
canny 边缘检测详细过程
"""

image = img.mean(axis=-1)

# 高斯平滑
def gaussian_detail():
	sigma = 0.5
	dim = 5
	gaussianFilter = np.zeros([dim, dim])
	tmp = [i - dim//2 for i in range(dim)]
	n1 = 1/(2*math.pi*sigma**2)
	n2 = -1/(2*sigma**2)

	for i in range(dim):
		for j in range(dim):
			gaussianFilter[i, j] = n1*math.exp(n2*(tmp[i]**2 + tmp[j]**2))
	gaussianFilter = gaussianFilter / gaussianFilter.sum()
	dx, dy = image.shape
	img_new = np.zeros(image.shape)
	tmp = dim//2
	img_pad = np.pad(image, ((tmp, tmp), (tmp, tmp)), 'constant')
	for i in range(dx):
		for j in range(dy):
			img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * gaussianFilter)
	plt.figure(1)
	plt.imshow(img_new.astype(np.uint8), cmap='gray')
	plt.axis('off')

	return img_new

# 2、求梯度。
def sobelKernel(img_new):
	sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	rows, cols = img_new.shape[:2]
	img_x = np.zeros(img_new.shape)
	img_y = np.zeros(img_new.shape)
	img_td = np.zeros(img_new.shape)
	img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
	for i in range(rows):
		for j in range(cols):
			img_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_x)
			img_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_y)
			img_td[i, j] = np.sqrt(img_x[i, j]**2 + img_y[i, j]**2)
	img_x[img_x == 0] = 0.00000001
	tan = img_y / img_x

	plt.figure(2)
	plt.imshow(img_td.astype(np.uint8), cmap='gray')
	plt.axis('off')

	return img_td, tan

# 3、非极大值抑制
def noMaxRepress(img_td, tan):
	img_yizhi = np.zeros(img_td.shape)
	dx, dy = img_td.shape[:2]
	for i in range(1, dx - 1):
		for j in range(1, dy - 1):
			flag = True
			temp = img_td[i-1:i+2, j-1:j+2]
			if tan[i, j] <= -1:  # 90度到135度
				num1 = (temp[0, 1] - temp[0, 0]) / tan[i, j] + temp[0, 1]
				num2 = (temp[2, 1] - temp[2, 2]) / tan[i, j] + temp[2, 1]
				if not (img_td[i, j] > num1 and img_td[i, j] > num2):
					flag = False
			elif tan[i, j] >= 1: # 45度到90度
				num1 = (temp[0, 2] - temp[0, 1]) / tan[i, j] + temp[0, 1]
				num2 = (temp[2, 0] - temp[2, 1]) / tan[i, j] + temp[2, 1]
				if not (img_td[i, j] > num1 and img_td[i, j] > num2):
					flag = False
			elif tan[i, j] > 0:  # 0度到90度
				num1 = (temp[0, 2] - temp[1, 2]) * tan[i, j] + temp[1, 2]
				num2 = (temp[2, 0] - temp[1, 0]) * tan[i, j] + temp[1, 0]
				if not (img_td[i, j] > num1 and img_td[i, j] > num2):
					flag = False
			elif tan[i, j] < 0:  # 135度到180度
				num1 = (temp[1, 0] - temp[0, 0]) * tan[i, j] + temp[1, 0]
				num2 = (temp[1, 2] - temp[2, 2]) * tan[i, j] + temp[1, 2]
				if not (img_td[i, j] > num1 and img_td[i, j] < num2):
					flag = False
			if flag:
				img_yizhi[i, j] = img_td[i, j]
	plt.figure(3)
	plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
	plt.axis('off')
	return img_yizhi

# 4、双阈值检测，连接边缘
def doubleThreshold(img_yizhi, img_td):
	lower_boundary = img_td.mean() * 0.5
	high_boundary = lower_boundary * 3
	zhan = []
	rows, cols = img_yizhi.shape[:2]
	for i in range(1, rows-1):
		for j in range(1, cols-1):
			if img_yizhi[i, j] >= high_boundary:
				img_yizhi[i, j] = 255
				zhan.append([i, j])
			elif img_yizhi[i, j] <= lower_boundary:
				img_yizhi[i, j] = 0

	while not len(zhan) == 0:
		temp_1, temp_2 = zhan.pop()
		a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
		if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
			img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
			zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
		if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
			img_yizhi[temp_1 - 1, temp_2] = 255
			zhan.append([temp_1 - 1, temp_2])
		if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
			img_yizhi[temp_1 - 1, temp_2 + 1] = 255
			zhan.append([temp_1 - 1, temp_2 + 1])
		if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
			img_yizhi[temp_1, temp_2 - 1] = 255
			zhan.append([temp_1, temp_2 - 1])
		if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
			img_yizhi[temp_1, temp_2 + 1] = 255
			zhan.append([temp_1, temp_2 + 1])
		if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
			img_yizhi[temp_1 + 1, temp_2 - 1] = 255
			zhan.append([temp_1 + 1, temp_2 - 1])
		if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
			img_yizhi[temp_1 + 1, temp_2] = 255
			zhan.append([temp_1 + 1, temp_2])
		if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
			img_yizhi[temp_1 + 1, temp_2 + 1] = 255
			zhan.append([temp_1 + 1, temp_2 + 1])

	for i in range(img_yizhi.shape[0]):
		for j in range(img_yizhi.shape[1]):
			if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
				img_yizhi[i, j] = 0

	# 绘图
	plt.figure(4)
	plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
	plt.axis('off')
	plt.show()


if __name__ == '__main__':

    # cannyImg()  # 普通 canny 边缘检测
    # canny_track()  # 可调节 canny 边缘检测

	"""
	canny 边缘检测详细过程
	"""
	img_new = gaussian_detail()
	img_td, tan = sobelKernel(img_new)
	img_yizhi = noMaxRepress(img_td, tan)
	doubleThreshold(img_yizhi, img_td)

if cv2.waitKey(0) == 27:
	cv2.destroyAllWindows()
