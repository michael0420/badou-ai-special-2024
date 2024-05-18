import random
import cv2

def GaussNoise(img, sigma, mean, snr):
    gauss_img = img
    s = int(img.shape[0] * img.shape[1] * snr)
    for i in range(s):
        x = random.randint(0, img.shape[0] - 1)
        y = random.randint(0, img.shape[1] - 1)

        gauss_img[x,y] = img[x,y] + random.gauss(mean, sigma)

        if gauss_img[x,y] > 255:
            gauss_img[x,y] = 255
        elif gauss_img[x,y] < 0:
            gauss_img[x,y] = 0
    return gauss_img



if __name__ == '__main__':
    img = cv2.imread("lenna.png",0)
    cv2.imshow("原图", img)
    gauss_img = GaussNoise(img, 2, 4, 0.5)
    cv2.imshow("添加高斯噪声后的图像", gauss_img)
    cv2.waitKey()


