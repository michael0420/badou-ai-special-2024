import cv2
import numpy as np

image = cv2.imread('photo1.jpg')

pts1 = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
pts2 = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

M = cv2.getPerspectiveTransform(pts1, pts2)

print(M)
transformed_image = cv2.warpPerspective(image, M, (337, 488))

# 显示原图和变换后的图像
cv2.imshow('Original', image)
cv2.imshow('Transformed', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()