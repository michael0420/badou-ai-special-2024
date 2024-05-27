import cv2

imread = cv2.imread("lenna.png")
color = cv2.cvtColor(imread, cv2.COLOR_BGR2GRAY)

create_sift = cv2.xfeatures2d.SIFT_create()
key, key_describe = create_sift.detectAndCompute(color, None)
img = cv2.drawKeypoints(image=imread, outImage=imread, keypoints=key,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

cv2.imshow("sift_describe", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
