import cv2
import numpy as np

img = cv2.imread('photo.jpg')

result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)

'''
import cv2 和 import numpy as np：导入OpenCV库和NumPy库。
img = cv2.imread('photo.jpg')：读取名为 "photo.jpg" 的图像文件。
result3 = img.copy()：创建原始图像的副本。
src 和 dst：定义了两个四边形区域的四个顶点的坐标，分别表示原始图像中待变换的区域和变换后的目标区域。
m = cv2.getPerspectiveTransform(src, dst)：通过输入源图像和目标图像的对应四个顶点的坐标，计算透视变换矩阵。
print("warpMatrix:") 和 print(m)：打印输出计算得到的透视变换矩阵。
result = cv2.warpPerspective(result3, m, (337, 488))：利用透视变换矩阵 m 对原始图像进行透视变换，将原始图像中的待变换区域映射到目标区域中，并指定输出图像的大小为 (337, 488)。
cv2.imshow("src", img)：显示原始图像。
cv2.imshow("result", result)：显示经过透视变换后的结果图像。
cv2.waitKey(0)：等待用户按下键盘上的任意键，然后关闭图像窗口。
这段代码的核心部分是通过 cv2.getPerspectiveTransform() 函数计算透视变换矩阵，然后利用 cv2.warpPerspective() 函数对图像进行实际的透视变换操作。

'''