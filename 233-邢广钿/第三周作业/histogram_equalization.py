import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 直方图均衡化
dst = cv2.equalizeHist(gray)

# 生成直方图(hist是长度为 256 的数组) arg：灰度图、通道、指定计算直方图时的掩模、直方图大小、像素值的范围
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
print(hist)
plt.figure()
# dst.ravel():均衡化后的图像数据展平为一维数组
plt.hist(dst.ravel(), 256)
plt.show()


# 均衡化三个通道的效果
(b,g,r) = cv2.split(img)
bh =cv2.equalizeHist(b)
gh =cv2.equalizeHist(g)
rh =cv2.equalizeHist(r)

result = cv2.merge((bh,gh,rh))
cv2.imshow("bgr_equalizehist",result)
cv2.waitKey(0)

# 均衡化GBR其中一个通道的效果
result1 = cv2.merge((bh,g,r))
result2 = cv2.merge((b,gh,r))
result3 = cv2.merge((b,g,rh))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
result1 = cv2.cvtColor(result1,cv2.COLOR_BGR2RGB)
result2 = cv2.cvtColor(result2,cv2.COLOR_BGR2RGB)
result3 = cv2.cvtColor(result3,cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(result1)
plt.subplot(223)
plt.imshow(result2)
plt.subplot(224)
plt.imshow(result3)
plt.show()

