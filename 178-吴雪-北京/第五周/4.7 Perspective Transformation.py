"""
透视变换 Perspective Transformation
透视变换本质：从一张图变为另一张图（两张图内容不变）
"""
import cv2
import numpy as np

img = cv2.imread('E:/Desktop/jianli/photo1.jpg')
result3 = img.copy()
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# 生成透视变换矩阵；进行透视变换        透视变换本质：从一张图变为另一张图（两张图内容不变）
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(img, m, (337, 488))  # 输出图像的大小(337, 488)这个是自己定义的
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
