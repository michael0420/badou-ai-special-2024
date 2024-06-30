import cv2

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints,descriptor = sift.detectAndCompute(gray, None)  #查找和计算

#画出关键点
img = cv2.drawKeypoints(image=img, keypoints=keypoints, outImage=img, color=(51, 163, 236),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT', img)
cv2.waitKey(0)
cv2.destroyAllWindows()