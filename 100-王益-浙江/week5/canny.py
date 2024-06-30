import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 不使用cv库函数，自己实现canny算法


def gauss_blur(original_img, sigma=1.0, ksize=5):
    img_blur = np.zeros(original_img.shape, np.float32)
    gauss_filter = np.zeros((ksize, ksize), np.float32)
    center = ksize // 2
    n1 = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(ksize):
        for j in range(ksize):
            gauss_filter[i, j] = n1 * math.exp(n2 * ((i - center) ** 2 + (j - center) ** 2))
    gauss_filter = gauss_filter / gauss_filter.sum()
    height, width = original_img.shape[:2]
    img_pad = np.pad(original_img, ((center, center), (center, center)), 'constant')
    for r in range(height):
        for c in range(width):
            img_blur[r, c] = np.sum(img_pad[r:r + ksize, c:c + ksize] * gauss_filter)
    plt.figure(1)
    plt.imshow(img_blur.astype(np.uint8), cmap='gray')
    plt.axis('off')
    return img_blur


def sobel_filter(img_blur, ksize=3):
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    img_pad = np.pad(img_blur, ((1, 1), (1, 1)), 'constant')
    width, height = img_blur.shape[:2]
    img_sobel = np.zeros(img_blur.shape)
    img_sobel_X = np.zeros(img_pad.shape)
    img_sobel_Y = np.zeros(img_pad.shape)
    for r in range(1, height + 1):
        for c in range(1, width + 1):
            img_sobel_X[r, c] = np.abs(
                np.sum(img_pad[r - ksize // 2:r + ksize // 2 + 1, c - ksize // 2:c + ksize // 2 + 1] * sobelX))
            img_sobel_Y[r, c] = np.abs(
                np.sum(img_pad[r - ksize // 2:r + ksize // 2 + 1, c - ksize // 2:c + ksize // 2 + 1] * sobelY))
            img_sobel[r - 1, c - 1] = np.sqrt(img_sobel_X[r, c] ** 2 + img_sobel_Y[r, c] ** 2)
    img_sobel_X[img_sobel_X == 0] = 0.0000001
    tan = img_sobel_Y / img_sobel_X
    plt.figure(2)
    plt.axis('off')
    plt.imshow(img_sobel.astype(np.uint8), cmap='gray')
    return img_sobel, tan


def is_max(pix0, a1, b1, a2, b2, tan):
    pix1 = (a1 - b1) * tan + b1
    pix2 = (a2 - b2) * tan + b2
    if pix0 > pix1 and pix0 > pix2:
        return pix0
    else:
        return 0


# 非极大值抑制
def non_max_suppression(img_sobel, tan):
    width, height = img_sobel.shape[:2]
    img_nms = np.zeros(img_sobel.shape, np.float32)
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            pix0 = img_sobel[r, c]
            if 0 < tan[r, c] <= 1:
                img_nms[r, c] = is_max(pix0, img_sobel[r - 1, c + 1], img_sobel[r, c + 1], img_sobel[r + 1, c - 1],
                                       img_sobel[r, c - 1], tan[r, c])
            elif -1 <= tan[r, c] <= 0:
                img_nms[r, c] = is_max(pix0, img_sobel[r + 1, c + 1], img_sobel[r, c + 1], img_sobel[r - 1, c - 1],
                                       img_sobel[r, c - 1], tan[r, c])
            elif tan[r, c] > 1:
                img_nms[r, c] = is_max(pix0, img_sobel[r - 1, c + 1], img_sobel[r - 1, c], img_sobel[r + 1, c - 1],
                                       img_sobel[r + 1, c], 1 / tan[r, c])
            elif tan[r, c] < -1:
                img_nms[r, c] = is_max(pix0, img_sobel[r + 1, c + 1], img_sobel[r + 1, c], img_sobel[r - 1, c - 1],
                                       img_sobel[r - 1, c], 1 / tan[r, c])

    plt.figure(3)
    plt.axis('off')
    plt.imshow(img_nms.astype(np.uint8), cmap='gray')
    return img_nms


# 双阈值检测
def binary_boundary_test(img):
    low_bound = 0.1 * (np.max(img) + np.min(img))
    high_bound = 0.2 * (np.max(img) + np.min(img))
    stack = []
    width, height = img.shape[:2]

    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if img[r, c] > high_bound:
                img[r, c] = 255
                stack.append((r, c))
            elif img[r, c] < low_bound:
                img[r, c] = 0

    vector_x = [0, 1, 0, -1, 1, -1, 1, -1]
    vector_y = [1, 0, -1, 0, 1, 1, -1, -1]
    while stack:
        r, c = stack.pop()
        for i in range(8):
            x = r + vector_x[i]
            y = c + vector_y[i]
            if high_bound > img[x, y] > low_bound:
                img[x, y] = 255
                stack.append((x, y))

    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if img[r, c] != 255 and img[r, c] != 0:
                img[r, c] = 0

    plt.figure(4)
    plt.axis('off')
    plt.imshow(img.astype(np.uint8), cmap='gray')
    return img


file_path = 'img/lenna.png'
img = cv2.imread(file_path).mean(axis=-1)
img_blur = gauss_blur(img, 1, 5)
img_sobel, tan = sobel_filter(img_blur)
img_nms = non_max_suppression(img_sobel, tan)
img_canny = binary_boundary_test(img_nms)
plt.show()
