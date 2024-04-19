import cv2
# 三通道均衡化,读取图片加载彩色照
png = cv2.imread('lenna.png', 1)
# cv2.imshow("src", png)
(blue, gray, read) = cv2.split(png)
bh = cv2.equalizeHist(blue)
gh = cv2.equalizeHist(gray)
rh = cv2.equalizeHist(read)
# 合并通道
L = (bh, gh, rh)
target_png = cv2.merge(L)
cv2.imshow("target", target_png)
cv2.waitKey(0)