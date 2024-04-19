import cv2
img = cv2.imread("lenna.png")
cv2.imshow("src", img)

#彩色图片均衡化
(b, g, r) = cv2.split(img)                    #彩色图像均衡化,需要分解通道 对每一个通道均衡化
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))              #合并每一个通道
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)

#灰度图片均衡化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #灰度化
dst = cv2.equalizeHist(gray)                  #均衡函数实现均衡化
cv2.imshow("dst_rgb", dst)
cv2.imshow("image_gray",gray)
cv2.waitKey(0)