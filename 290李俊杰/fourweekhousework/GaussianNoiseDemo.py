'''
【第四周作业】

作业：1.实现高斯噪声 2.实现椒盐噪声 3.实现PCA  4.拓展：证明中心化协方差矩阵公式

'''


'''
一个正常的高斯采样分布公式, 得到输出像素Pout. Pout = Pin + random.gauss
其中random.gauss是通过sigma和mean来生成符合高斯分布的随机数。
给一副数字图像加上高斯噪声的处理顺序如下：
a. 输入参数sigma 和 mean
b. 生成高斯随机数
d. 根据输入像素计算出输出像素
e. 重新将像素值放缩在[0 ~ 255]之间
f. 循环所有像素
g. 输出图像
'''
# 1.实现高斯噪声
#模拟的高斯噪声分布本身就是随机出现且数量不定的。
import  random
import  cv2
from skimage import util
# 以灰度图方式读图
img=cv2.imread("lenna.png",0)
cv2.imshow("img show",img)
sigma=3
mu=5
h,w=img.shape
print(img)
# 随机对图片上的像素值添加高斯噪声
# 确定高斯噪音所占比例
noisenum=int(0.7*h*w+0.5)
for noisenum in range(noisenum):
        # 高斯分布的随机数生成使用random.gauss(mu,sigma)
        raX=random.randint(0,h-1)
        raY = random.randint(0, w - 1)
        img[raX,raY]=img[raX,raY]+random.gauss(mu,sigma)
        if img[raX,raY]>255:
            img[raX, raY]=255
        elif img[raX,raY]<0:
            img[raX, raY] = 0
print(img)
cv2.imshow("gauss img show",img)
cv2.waitKey(0)

#可以使用封装的函数实现高斯噪声util.random_noise(img,mode='poisson')
# mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
# var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01（注：不是标准差）
# img1 = cv2.imread("lenna.png", 0)
# img1=util.random_noise(img,mode='gaussian',mean=0.2,var=0.2)
# cv2.imshow("saltpper img show",img1)
# cv2.waitKey(0)