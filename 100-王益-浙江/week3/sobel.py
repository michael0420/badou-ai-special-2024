import cv2
import numpy as np


# 手动实现Sobel
def apply_sobel(img_gray):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    height, width = img_gray.shape[:2]

    output_x = np.zeros([width, height], dtype=np.int16)
    output_y = np.zeros([width, height], dtype=np.int16)

    for i in range(height - 2):
        for j in range(width - 2):
            output_x[i, j] = np.sum(img_gray[i:i + 3, j:j + 3] * sobel_x)
            output_y[i, j] = np.sum(img_gray[i:i + 3, j:j + 3] * sobel_y)

    output_x = cv2.convertScaleAbs(output_x)
    output_y = cv2.convertScaleAbs(output_y)
    output = cv2.addWeighted(output_x, 0.5, output_y, 0.5, 0)
    return output;


# 使用OpenCV的Sobel函数
def apply_sobel_cv2(img_gray):
    output_x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
    output_y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)

    output_x = cv2.convertScaleAbs(output_x)
    output_y = cv2.convertScaleAbs(output_y)
    output = cv2.addWeighted(output_x, 0.5, output_y, 0.5, 0)
    return output


img = cv2.imread('img/lenna.png', 0)
output = apply_sobel(img)
cv2.imshow('Sobel', output)
output_cv2 = apply_sobel_cv2(img)
cv2.imshow('Sobel_cv2', output_cv2)
cv2.waitKey(0)
