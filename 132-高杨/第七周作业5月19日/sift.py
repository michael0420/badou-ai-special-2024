import cv2




img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)


#创建
sift = cv2.SIFT_create()
keypoints,descriptors=sift.detectAndCompute(gray,None)
# 显示关键点检测后的结果
img =cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51,163,236))

cv2.imshow('sift_keypoints',img)
cv2.waitKey(0)
cv2.destroyAllWindows()






