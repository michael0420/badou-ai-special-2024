import  cv2
import numpy as np
from matplotlib import pyplot as plt

#
# #灰度图直方图均衡化
# image1 = cv2.imread("yangmi.jpg",1)
# image = cv2.resize(image1,(360, 640))
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow("image_gray", gray)
#
# dst = cv2.equalizeHist(gray)
#
# hist = cv2.calcHist([dst],[0],None,[256],[0,256])
#
# plt.figure()
# plt.hist(dst.ravel(),256)
# plt.show()
#
# cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
# cv2.waitKey(0)



# 彩色图像直方图均衡化
image1 = cv2.imread('yangmi.jpg')  # 读图
img = cv2.resize(image1, (360, 640))  # 图片缩放

(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))


plt.subplot(221), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('junhenghuaqian')

color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.subplot(222), plt.plot(histr, color=col),
    plt.xlim([0, 256]), plt.title('Histogram')

plt.subplot(223), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('junhenghuahou')

color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([result], [i], None, [256], [0, 256])
    plt.subplot(224), plt.plot(histr, color=col),
    plt.xlim([0, 256]), plt.title('Histogram')
plt.show()


cv2.imshow("Histogram Equalization", np.hstack([img, result]))
cv2.waitKey(0)
cv2.destroyAllWindows()



