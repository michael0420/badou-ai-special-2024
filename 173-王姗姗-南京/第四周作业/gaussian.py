# 随机生成符合高斯分布的随机数
import cv2
import random
from skimage import util


# 生成高斯噪声的方法 src:图片路径 means,sigma：生成高斯随机数的两个参数，percetage:百分比
def gaussianNoise(src, means, sigma, rate):
    # 噪音图片
    noiseImg = src
    # 噪音点数量
    noiseNum = int(rate * src.shape[0] * src.shape[1])

    for i in range(noiseNum):
        # 每次取出一个随机点
        # 把一张图片的像素用行和列表示的话，randX代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机数
        # 高斯噪声图片边缘不做处理，所以-1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # 此处在原有像素灰度值上加上随机数
        noiseImg[randX, randY] = noiseImg[randX, randY] + random.gauss(means, sigma)
        # 若灰度值小于0 的则强制为0，若灰度值大于255则强制255
        if noiseImg[randX, randY] < 0:
            noiseImg[randX, randY] = 0
        elif noiseImg[randX, randY] > 255:
            noiseImg[randX, randY] = 255
    return noiseImg


img = cv2.imread('lenna.png', 0)
img1 = gaussianNoise(img, 2, 4, 0.8)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用工具类转高斯噪声
noise_gas_img = util.random_noise(img, mode='gaussian')

cv2.imshow('lenna_gaussian', img1)
cv2.imshow('source', img2)
cv2.imshow('util_gaussian', noise_gas_img)
cv2.waitKey(0)
