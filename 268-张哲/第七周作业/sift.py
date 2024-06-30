import cv2

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints,descript = sift.detectAndCompute(gray,None)
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。

img = cv2.drawKeypoints(image=img,keypoints = keypoints,outImage=img,color=(51,163,236),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('soft_keypoints',img)
cv2.waitKey(0)
cv2.destroyAllWindows()