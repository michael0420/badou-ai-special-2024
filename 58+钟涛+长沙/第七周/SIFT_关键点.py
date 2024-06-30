import cv2

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#调用sift算法
sift = cv2.xfeatures2d.SIFT_create()

#sift 检测和计算
# 检测关键点并计算描述符
keypoints, descriptor = sift.detectAndCompute(gray, None)

#画图
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("sift",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
