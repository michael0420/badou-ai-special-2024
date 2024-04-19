import cv2
import numpy as np
import matplotlib.pyplot as plt
import grayscale


def convert_to_colorless_image(img_path):
    img_gray = grayscale.convert_to_gray_image_pyplot(img_path)
    clean_color(img_gray)
    return img_gray


def clean_color(img_gray):
    rows, cols = img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if img_gray[i, j] <= 0.5:
                img_gray[i, j] = 0
            else:
                img_gray[i, j] = 1


def convert_to_colorless_image_np(img_path):
    img = plt.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_binary = np.where(img_gray >= 0.5, 1, 0)
    return img_binary
