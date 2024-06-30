import cv2

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

# 在灰度图像上检测 SIFT 特征点，并计算它们的描述符
keypoints, descriptor = sift.detectAndCompute(gray, None)

#img=cv2.drawKeypoints(gray,keypoints,img)

# 在图像上绘制检测到的 SIFT 特征点
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()