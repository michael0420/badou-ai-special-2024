import cv2
import gray
from skimage.color import rgb2gray


def generate_binary():
    image = cv2.imread("lenna.png")
    image_gray = rgb2gray(image)
    height = image_gray.shape[0]
    width = image_gray.shape[1]
    for h in range(height):
        for w in range(width):
            if image_gray[h, w] <= 0.5:
                image_gray[h, w] = 0
            else:
                image_gray[h, w] = 1
    return image_gray


if __name__ == '__main__':
    cv2.imshow("show binary", generate_binary())
    cv2.waitKey(0)

