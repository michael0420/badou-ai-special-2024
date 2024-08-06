import cv2
import numpy as np

# 读取图像
img = cv2.imread('photo1.jpg')

# 复制图像，防止对原图像进行修改
result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''

# 定义源顶点坐标（原图像中需要进行透视变换的四个顶点）
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])

# 定义目标顶点坐标（变换后图像中的顶点坐标）
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# 打印原图像的形状
print(img.shape)

# 生成透视变换矩阵
m = cv2.getPerspectiveTransform(src, dst)

# 打印透视变换矩阵
print("warpMatrix:")
print(m)

# 使用透视变换矩阵对图像进行透视变换
result = cv2.warpPerspective(result3, m, (337, 488))

# 显示原图像
cv2.imshow("src", img)

# 显示透视变换后的图像
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
