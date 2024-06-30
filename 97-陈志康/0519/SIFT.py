import cv2

#  conda(pip) install opencv-python==3.4.2.16
#  conda(pip) install opencv-contrib-python==3.4.2.16

# 读图
img = cv2.imread("../0327/lenna.png")
# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 声明
sift = cv2.SIFT_create()
# 输入数据
keypoints, descriptor = sift.detectAndCompute(gray, None)
# 关键点     描述符

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

# img=cv2.drawKeypoints(gray,keypoints,img)

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
