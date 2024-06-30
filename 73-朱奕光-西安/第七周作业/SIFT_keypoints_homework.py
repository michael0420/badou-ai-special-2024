import cv2

img = cv2.imread('caipiao2_resize.jpg')
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img, None)
dst = cv2.drawKeypoints(img, kp1, img,
                        color=(199, 237, 204),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('dst', dst)
cv2.waitKey(0)
