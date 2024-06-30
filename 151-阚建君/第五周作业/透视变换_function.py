import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('random.jpg')
target_size = (615,915)
# 调整图片大小
resized_image = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
result3 = resized_image.copy()

# 定点
# plt.figure(1)
# plt.imshow(result3)
# plt.axis('off')  # 关闭坐标刻度值
# plt.show()

src = np.float32([[10, 432], [196, 231], [409, 807], [602, 525]])
dst = np.float32([[0, 0], [350, 0], [0, 600], [350, 600]])
print(resized_image.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (350, 600))
cv2.imshow("src", resized_image)
cv2.imshow("result", result)
cv2.waitKey(0)
