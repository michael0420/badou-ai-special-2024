import cv2
import random
from skimage import util

def noise_gauss(img, noise_rate, means, sigma):
    """
    :param img: 传入要加高斯噪声的图片
    :param noise_rate: 加噪声的比例
    :param means,sigma: random.gauss的参数
    :return:输出加高斯噪声后的图片
    """
    w, h = img.shape
    img_noise = img.copy()
    for i in range(int(w*h*noise_rate)):
        # 注意边界取值为w-1,h-1
        randX = random.randint(0, h-1)
        randY = random.randint(0, w-1)
        img_noise[randX, randY] = img[randX, randY] + random.gauss(means, sigma)
        # 修改过大过小值
        if img_noise[randX, randY] > 255:
            img_noise[randX, randY] = 255
        elif img_noise[randX, randY] < 0:
            img_noise[randX, randY] = 0
    return img_noise

if __name__ == '__main__':
    img = cv2.imread('lenna.png', 0)
    cv2.imshow('source', img)
    img_noise = noise_gauss(img, 0.8, 4, 2)
    cv2.imshow('noise_gauss', img_noise)
    cv2.waitKey(0)

