import cv2
import random

def JiaoYanNoise(src,percetage):
    NoiseImg = src                                              # 拿到要处理的图片
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])     # 计算要处理的像素点数
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX代表随机生成的行，ranY代表随机生成的列
        # random.random()生成随机浮点数
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        # 随机取到一个像素，一般几率为0，一般几率为255
        if random.random() <= 0.5:
            NoiseImg[randX,randY] = 0
        else :
            NoiseImg[randX,randY] = 255
    return NoiseImg

img = cv2.imread("lenna.png",1)                   # 原图的灰度图    0是灰度，1是彩色
img_JiaoYan = JiaoYanNoise(img,0.2)               # 椒盐噪声图
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # 灰度图
cv2.imshow("img",img)
cv2.imshow("JiaoYanNoise",img_JiaoYan)
cv2.waitKey()