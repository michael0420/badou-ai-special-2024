import numpy as np
import cv2
import random
import copy
from skimage import util

#高斯噪声
def func1(img,mean,sigma,proportion):
    h,w=img.shape
    result=copy.deepcopy(img)
    #对proportion*h*w的点进行加噪处理，可能会有重复的点，因为后续的x，y的取值可能会和之前轮数的取值相同
    for i in range(int(proportion*h*w)):
        randx=random.randint(0,h-1)
        randy=random.randint(0,w-1)
        result[randx,randy]=random.gauss(mean,sigma)+img[randx,randy]
        result[randx,randy]=max(0,result[randx,randy])
        result[randx,randy]=min(255,result[randx,randy])
    return  result

#椒盐噪声
def func2(img,proportion):
    h,w=img.shape
    result=copy.deepcopy(img)
    for i in range(int(proportion*h*w)):
        randx=random.randint(0,h-1)
        randy=random.randint(0,w-1)
        result[randx,randy]=random.randint(0,1)*255
    return  result

#图像增强（点处理，线性变换）
def func3(img):
    h,w=img.shape
    result=copy.deepcopy(img)
    for i in range(h):
        for j in range(w):
            result[i,j]=min(255,result[i,j]*2+10)
    return result

if __name__=='__main__':
    #加噪，高斯噪声和椒盐噪声
    img=cv2.imread("lenna.png")
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img1=func1(gray_img,2,4,0.8)                #img1为高斯噪声
    img2=func2(gray_img,0.05)                   #img2为椒盐噪声
    cv2.imshow("nosie",np.hstack([gray_img,img1,img2]))
    cv2.waitKey(0)

    #滤波，降噪
    blur1=cv2.blur(img1,(3,3))                  #对高斯噪声进行均值与中值滤波,cv2.blur()均值滤波  cv2.medianBlur()中值滤波
    medianBlur1=cv2.medianBlur(img1,3)

    blur2=cv2.blur(img2,(3,3))                  #对椒盐噪声进行均值与中值滤波
    medianBlur2=cv2.medianBlur(img2,3)

    cv2.imshow("guass_noise",np.hstack([img1,blur1,medianBlur1]))
    cv2.imshow("salt_pepper_noise",np.hstack([img2,blur2,medianBlur2]))
    cv2.waitKey(0)

    #图像增强
    contrast=func3(gray_img)
    cv2.imshow("image_enhancement",np.hstack([gray_img,contrast]))
    cv2.waitKey(0)

    #简洁版加噪
    img = cv2.imread("lenna.png")
    noise_gs_img = util.random_noise(img, mode='gaussian')
    '''
    def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
    功能：为浮点型图片添加各种随机噪声
    参数：
    image：输入图片（将会被转换成浮点型），ndarray型
    mode： 可选择，str型，表示要添加的噪声类型
    	gaussian：高斯噪声
    	localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差。
    	poisson：泊松噪声
    	salt：盐噪声，随机将像素值变成1
    	pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
    	s&p：椒盐噪声
    	speckle：均匀噪声（均值mean方差variance），out=image+n*image
    seed： 可选的，int型，如果选择的话，在生成噪声前会先设置随机种子以避免伪随机
    clip： 可选的，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片的数据裁剪到合适范围内。如果是False，则输出矩阵的值可能会超出[-1,1]
    mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
    var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01（注：不是标准差）
    local_vars：可选的，ndarry型，用于定义每个像素点的局部方差，在localvar中使用
    amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
    salt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
    --------
    返回值：ndarry型，且值在[0,1]或者[-1,1]之间，取决于是否是有符号数
    '''


