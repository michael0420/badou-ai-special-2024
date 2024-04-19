import numpy as np
import cv2
import matplotlib.pyplot as plt

def sobel1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 求取x方向和y方向的梯度
    sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    # 分别将x和y方向的梯度取绝对值并相加
    sobelxy = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sobelxy, cmap='gray')
    plt.title('Sobel Sobel')
    plt.axis('off')


def sobel2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 定义自定义的卷积核
    kernel1 = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    kernel2 = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    # 使用 filter2D 函数进行卷积操作
    sobelx = cv2.filter2D(gray, -1, kernel1)
    sobely = cv2.filter2D(gray, -1, kernel2)
    sobelxy = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sobelxy, cmap='gray')
    plt.title('Sobel Sobel')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    img = cv2.imread("0216.jpg")
    sobel1(img)
    sobel2(img)
