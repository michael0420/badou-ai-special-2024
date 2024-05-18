# Canny 边缘检测

1. **噪声去除**：
   使用高斯滤波器平滑图像，以去除噪声。

   ```python
   import cv2
   import numpy as np

   image = cv2.imread('path_to_image')
   blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
   ```
2. **梯度计算**：
   计算图像中每个像素点的梯度强度和方向。常用 Sobel 算子来计算。
   ```python
    gx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * (180 / np.pi) % 180
   ```
3. **非极大值抑制**：
   在梯度方向上检查像素，仅保留局部最大梯度值的点。
4. **双阈值检测**：
   在梯度方向上检查像素，仅保留局部最大梯度值的点。
   ```python
    edges = cv2.Canny(blurred_image, 50, 150)
   ```
5. **边缘连接**：
通过抑制孤立的弱边缘最终确定图像中的边缘。

# 透视变换
透视变换是计算机视觉中处理图像以改变观看视角的一种技术。

1. **选择四个点**：
在原图像和目标图像中分别选取四个点。
```python
# 假设 pts_src 和 pts_dst 分别是原图像和目标图像中的点
pts_src = np.array([[0, 0], [400, 0], [400, 400], [0, 400]])
pts_dst = np.array([[50, 50], [350, 50], [350, 350], [50, 350]])
```
2. **计算变换矩阵**：
使用所选的点，通过解算方程组来计算变换矩阵。
```python
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
```
3. **应用变换**：
使用上步得到的变换矩阵，将原图像中的每个像素映射到新的位置。
```python
transformed_image = cv2.warpPerspective(image, matrix, (400, 400))
```
