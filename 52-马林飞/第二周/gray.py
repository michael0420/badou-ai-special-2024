import numpy as np
import cv2

# 灰度化
image = cv2.imread("lenna.png")
h, w = image.shape[:2]  # 获取图片的high和wide
imageGray = np.zeros([h, w], image.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = image[i, j]  # 取出当前high和wide中的BGR坐标
        imageGray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像

cv2.imshow("image show", image)
cv2.imshow("image show gray", imageGray)
cv2.waitKey()
cv2.destroyAllWindows()
