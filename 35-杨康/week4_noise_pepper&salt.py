import cv2
import random

def img_noise(img, noise_rate):
    """
    :param img: 传入要加噪声的图片
    :param noise_rate: 加噪声的比例
    :return:输出加噪声后的图片
    """
    w, h = img.shape
    img_noise = img.copy()
    for i in range(int(w*h*noise_rate)):
        # 注意边界取值为w-1,h-1
        randX = random.randint(0, h-1)
        randY = random.randint(0, w-1)
        # 椒噪声0，盐噪声255
        if random.random() <= 0.5:
            img_noise[randX, randY] = 255
        else:
            img_noise[randX, randY] = 0
    return img_noise

if __name__ == '__main__':
    img = cv2.imread('lenna.png', 0)
    cv2.imshow('source', img)
    img_noise = img_noise(img, 0.2)
    cv2.imshow('img_noise', img_noise)
    cv2.waitKey(0)

