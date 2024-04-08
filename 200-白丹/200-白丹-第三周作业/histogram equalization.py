# 直方图均衡化
import cv2

# 灰度图像直方图均衡化
img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst_img = cv2.equalizeHist(img_gray)
cv2.imshow("dst_img", dst_img)
cv2.waitKey()

# 彩色直方图均衡化
img = cv2.imread("lenna.png")
# 对每一个通道均衡化
(b,g,r) = cv2.split(img)  # 将多通道图像分解为单通道，注意BGR
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并为一个通道
dst_rgb = cv2.merge([bH, gH, rH])
cv2.imshow("dst_rgb", dst_rgb)
cv2.waitKey()
