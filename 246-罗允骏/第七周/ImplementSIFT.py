import cv2

# 读取图像
img = cv2.imread('lenna.png')

# 创建SIFT检测器对象
sift = cv2.SIFT_create()

# 使用detectAndCompute方法检测关键点并计算描述符
keypoints, descriptors = sift.detectAndCompute(img, None)

# 在图像上绘制关键点（可选）
output_img = cv2.drawKeypoints(img, keypoints, None)

# 显示图像
cv2.imshow('SIFT KeyPoints', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()