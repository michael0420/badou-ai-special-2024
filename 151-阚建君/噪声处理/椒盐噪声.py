import cv2
import random

def pepper_salt(source,percentage):
    noise_img = source
    noise_num = int(noise_img.shape[0] * noise_img.shape[1]*percentage)
    for i in range(noise_num):
        randomX = random.randint(0,source.shape[0]-1)   # 不能取到边界
        randomY = random.randint(0,source.shape[1]-1)   # 不能取到边界
        if random.random()<=0.5:
            noise_img[randomX,randomY] = 0
        else:
            noise_img[randomX, randomY] = 255
    return noise_img

if __name__ == '__main__':
    img=cv2.imread('lenna.png',0)
    img1=pepper_salt(img,0.5)
    img = cv2.imread('lenna.png')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('source',img2)
    cv2.imshow('lenna_PepperandSalt',img1)
    cv2.waitKey(0)

