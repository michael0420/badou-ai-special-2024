import cv2
import numpy as np

def generate_gray():
    image = cv2.imread("lenna.png")
    height = image.shape[0]
    width = image.shape[1]
    image_gray = np.zeros([height, width], image.dtype)
    for h in range(height):
        for w in range(width):
            pixel = image[h, w]
            image_gray[h, w] = int(pixel[0] * 0.11 + pixel[1] * 0.59 + pixel[2] * 0.3)
    return image_gray


if __name__ == '__main__':
    cv2.imshow("show gray",generate_gray())
    cv2.waitKey(0)

