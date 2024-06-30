#随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
import numpy as np  # 导入numpy库，用于科学计算
import cv2  # 导入opencv库，用于图像处理
import random  # 导入random库，用于生成随机数
from numpy import shape

'''
这段代码的主要功能是为输入的图像添加高斯噪声。具体来说,它包含以下几个步骤:

1. **导入必要的库**:代码首先导入了numpy、opencv和random库,用于数值计算、图像处理和生成随机数。

2. **定义添加高斯噪声的函数GaussianNoise()**:这个函数接收四个参数:
   - src: 输入的原始图像
   - means: 高斯分布的均值
   - sigma: 高斯分布的标准差
   - percetage: 需要添加噪声的像素点占总像素点的百分比

3. **在函数内部,遍历需要添加噪声的像素点**:
   - 计算需要添加噪声的像素点个数
   - 对于每个需要添加噪声的像素点,随机生成其行列坐标
   - 使用random.gauss()函数生成服从正态分布的随机数,并将其加到对应像素点的灰度值上
   - 对超出灰度值范围(0-255)的像素点进行截断处理

4. **读取原始图像**:代码读取名为'lenna.png'的图像文件,并将其转换为灰度图像。

5. **调用GaussianNoise()函数添加高斯噪声**:使用均值为2、标准差为4、噪声像素点占80%的参数调用该函数,得到添加噪声后的图像img1。

6. **显示原始图像和添加噪声后的图像**:代码使用OpenCV的imshow()函数分别显示原始灰度图像和添加噪声后的图像。

7. **等待用户按键退出**:代码使用cv2.waitKey(0)无限期等待用户按下任意键,以便观察显示的图像。

总的来说,这段代码实现了为输入图像添加高斯噪声的功能,可以用于模拟图像在传输或获取过程中受到噪声污染的情况,对图像处理和计算机视觉等领域有一定的应用价值。
'''

# 定义一个函数，用于为输入的图像添加高斯噪声
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg = src   # 复制输入的图像，避免修改原始图像
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])    # 计算需要添加高斯噪声的像素点百分比*行*列
    for i in range(NoiseNum):  # 循环遍历每个需要添加噪声的像素点
		# 每次取一个随机点
		# 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
		# 高斯噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0] - 1)  # 随机生成行坐标
        randY = random.randint(0, src.shape[1] - 1)  # 随机生成列坐标

        # 此处在原有像素灰度值上加上随机数
        # 为当前像素点添加服从正态分布的随机数作为噪声
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        '''
        `random.gauss(means, sigma)` 是Python中random模块的一个函数,用于生成服从正态分布(高斯分布)的随机数。它的运算过程如下:
        1. 首先,函数接收两个参数:
           - `means`(μ): 正态分布的均值
           - `sigma`(σ): 正态分布的标准差
        
        2. 函数根据以下公式计算正态分布的概率密度函数(PDF):
           
           f(x) = (1 / (σ * sqrt(2π))) * e^(-(x - μ)^2 / (2σ^2))
        
           其中:
           - x是随机变量的值
           - μ是均值
           - σ是标准差
           - π是圆周率
           - e是自然对数的底数
        
        3. 函数使用一种叫做"拒绝采样"(Rejection Sampling)的方法,从该概率密度函数中采样一个随机数x。
        
        4. 生成的随机数x就服从N(μ, σ^2)的正态分布。[1]
        
        因此,`random.gauss(means, sigma)`通过给定的均值means和标准差sigma参数,利用拒绝采样方法从正态分布的概率密度函数中采样出一个随机数。
        这个随机数的均值为means,标准差为sigma,符合正态分布的性质。
        
        需要注意的是,`random.gauss()`生成的是伪随机数,而不是真正的随机数。但在大多数应用场景下,这些伪随机数已经足够随机,可以满足需求。
        '''
        # print(random.gauss(means, sigma))

        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        # 如果添加噪声后的像素值小于0，则设为0；如果大于255，则设为255  随机分布的黑白噪声
        if  NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255

    return NoiseImg   # 返回添加噪声后的图像

img = cv2.imread('lenna.png', 0)  # 读取灰度图像，将其赋值给img  flag=0 灰度图显示
'''
在高斯模糊(Gaussian Blur)中,sigma是一个重要的参数,它决定了高斯核的大小和形状。具体来说,sigma有以下含义:
高斯核的大小
高斯核的大小由sigma决定。通常情况下,高斯核的大小设置为truncOuter = cvRound(sigma * 6 + 1)。这意味着,当sigma越大,高斯核的大小就越大。较大的高斯核可以产生更强的平滑效果,但同时也会增加计算量。
高斯分布的标准差
sigma实际上就是高斯分布的标准差。在高斯核中,离中心越远的像素点,其权重就越小。sigma决定了这种权重的衰减速率。较大的sigma意味着远离中心的像素点也会获得较高的权重,从而产生更多的模糊效果。
模糊程度
sigma的大小直接决定了模糊的程度。当sigma较小时,只有邻近的像素会对目标像素产生影响,从而产生较小的模糊效果。当sigma较大时,远离目标像素的像素也会对其产生影响,从而产生较大的模糊效果。
因此,在使用高斯模糊时,sigma是一个需要根据具体情况调整的重要参数。较小的sigma可以用于去除噪声,而较大的sigma可以用于图像的平滑处理。选择合适的sigma值对于获得理想的模糊效果非常重要。
'''

img1 = GaussianNoise(img, 2, 4, 0.8)  # 对图像img添加高斯噪声，均值为2，标准差为4，噪声占图像像素点的百分比为80%
cv2.imshow('lenna_GaussianNoise', img1)  # 显示添加高斯噪声后的图像
# cv2.imwrite('lenna_GaussianNoise.png', img1)  # 将添加高斯噪声后的图像保存为文件
cv2.waitKey(0)  # 等待键盘输入，参数0表示无限等待

img = cv2.imread('lenna.png')  # 重新读取原始彩色图像，将其赋值给img
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像，将其赋值给img2
cv2.imshow('source', img2)  # 显示原始灰度图像
cv2.waitKey(0)  # 等待键盘输入，参数0表示无限等待
