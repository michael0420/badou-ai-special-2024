import cv2

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(cv2.__version__)

'''
专利到期：SIFT的专利已经在2020年3月到期。专利保护通常持续20年，从专利申请之日算起。SIFT算法的专利申请日期是1999年，因此在2020年其专利保护期满，这使得SIFT成为了公有领域的技术，
任何人都可以自由地使用、分发和修改该算法而不必担心专利侵权。

OpenCV的更新：随着SIFT专利的到期，OpenCV项目决定将SIFT重新纳入主库。这意味着用户不再需要安装额外的opencv-contrib模块就可以使用SIFT。
'''

# sift = cv2.xfeatures2d.SIFT_create()

# OpenCV 4.4.0 之后的使用方法
sift = cv2.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

# img=cv2.drawKeypoints(gray,keypoints,img)

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()