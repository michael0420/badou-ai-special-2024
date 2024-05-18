import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4)
    sobel_x = cv2.Sobel(new_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(new_img, cv2.CV_64F, 0, 1, ksize=3)
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)
    sobel = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

    lower = sobel.mean() * 0.5
    high = lower * 3
    canny = cv2.Canny(gray, 100, 150)

    plt.subplot(231), plt.imshow(gray, "gray"), plt.title("Original")
    plt.subplot(232), plt.imshow(sobel_x, "gray"), plt.title("Sobel_x")
    plt.subplot(233), plt.imshow(sobel_y, "gray"), plt.title("Sobel_y")
    plt.subplot(234), plt.imshow(sobel, "gray"), plt.title("Sobel")
    plt.subplot(235), plt.imshow(laplace, "gray"), plt.title("Laplace")
    plt.subplot(236), plt.imshow(canny, "gray"), plt.title("Canny")
    plt.show()
