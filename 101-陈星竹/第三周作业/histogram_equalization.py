import cv2
import numpy as np
from matplotlib import pyplot as plt


'''
灰度图像
'''
img = cv2.imread('lenna.png')
#灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#均衡化前的直方图
#表示计算灰度图像 gray 的直方图，使用的通道是第 0 个通道（即灰度通道），直方图的大小为 256，像素值的范围为 0 到 255。
#None表示指定掩码图像，即不显示像素为xx的点，None就表示没有要掩盖的
# '''
# hist1 = cv2.calcHist([gray],[0],None,[256],[0,256])
# plt.figure()
# #dst.ravel() 将数组展平为一维数组，所有元素按行优先顺序排列成一个新的一维数组,256表示直方图的柱数
# plt.hist(hist1.ravel(),256)
# plt.show()
# '''

#直方图均衡化
dst = cv2.equalizeHist(gray)
hist2 = cv2.calcHist([dst],[0],None,[256],[0,256])
# 创建一个新的图形窗口
plt.figure()
#dst.ravel()展开后计算更快
plt.hist(dst.ravel(),256)
plt.show()

#图像对比
#使用 np.hstack() 函数水平堆叠两张图像
concatenated_img = np.hstack([gray,dst])
cv2.imshow("Histogram Equalization",concatenated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
#彩色图像,切割成三个通道分别直方图均衡化再合并

(b,g,r) = cv2.split(img)
#分成三个单通道，会默认展示成灰度图像，因此是灰色
# cv2.imshow("splited img",np.hstack([b,g,r]))

#分别进行直方图均衡
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)

#合并
dst_img = cv2.merge((bh,gh,rh))
cv2.imshow('dst_rgb and original_img',np.hstack([dst_img,img]))
cv2.waitKey(0)
'''