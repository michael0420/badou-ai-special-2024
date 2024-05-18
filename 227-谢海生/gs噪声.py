import numpy as np
import cv2
from numpy import shape
import random
'''
这个函数接受四个参数：src是原始图像，means是高斯噪声分布的平均值，sigma是标准差，percentage是应用噪声的百分比。
函数内部，首先创建了一个与src相同的副本，命名为NoiseImg。然后，通过将percentage乘以src图像的行数和列数来计算应该添加噪声的像素数量，
并将结果存储在NoiseNum中
'''
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):

#每次取一个随机点
#把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY 代表随机生成的列
#random.randint生成随机整数
#高斯噪声图片边缘不处理，故-1
     randX = random.randint(0,src.shape[0]-1)
     randY = random.randint(0,src.shape[1]-1)
#此处在原有像素灰度值上加上随机数
    NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
#若灰度值小于0则强制为0，若灰度值大于255则强制为255
    if NoiseImg[randX,randY]<0:
        NoiseImg[randX,randY]=0
    elif NoiseImg[randX,randY]>255:
        NoiseImg[randX,randY]=255
    return NoiseImg


img = cv2.imread("C:/Users/86188/Pictures/lenna.png",0)
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread("C:/Users/86188/Pictures/lenna.png")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imwrite('C:/Users/86188/Pictures/lenna_GaussianNoise.png',img1)
cv2.imshow('source',img2)
cv2.imshow('C:/Users/86188/Pictures/lenna_GaussianNoise',img1)
cv2.waitKey(0)

"""

    使用 cv2.imread 函数加载 "C:/Users/86188/Pictures/lenna.png" 图片，参数 0 表示以灰度模式加载。
    调用 GaussianNoise 函数向 img 图片添加高斯噪声，这里的参数意味着噪声的均值为 2，标准差为 4，且 80% 的像素将被噪声处理。
    再次使用 cv2.imread 加载同一张图片 lenna.png，但这次不指定颜色通道，因此会以彩色模式加载。
    使用 cv2.cvtColor 将彩色的 img 转换为灰度图 img2。
    使用 cv2.imshow 分别显示原始灰度图 img2 和添加了高斯噪声的灰度图 img1。
    使用 cv2.waitKey(0) 等待按键事件，这会使窗口保持打开状态，直到用户按键。

请注意，代码中的 GaussianNoise 函数必须已经定义并且实现正确，才能正常添加高斯噪声。另外，图片路径应确保正确无误，且程序应具有足够的权限访问该路径。



"""





