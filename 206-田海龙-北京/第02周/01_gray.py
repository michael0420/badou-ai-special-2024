
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

from Util import cv_imread

def img_gray_01(img,is_decimal=False):
    """
    灰度化，自己写方法
    is_decimal: 是否返回浮点数，True返回浮点数，False返回整数
    """
    #获取图片的high和wide
    h,w = img.shape[:2]
    #创建一张和当前图片大小一样的单通道图片
    
    dtype = img.dtype
    # 如果显示小数，需要设置dtype为float64，否则赋值为0-1小数后，值都变为0
    if(is_decimal):
        dtype=np.float64

    img_gray = np.zeros([h,w],dtype)

    # 复制对象，避免影响后续操作
    img_temp=img.copy()

    # 如果以小数表示，则除以255
    if(is_decimal):
        img_temp=img_temp/255.0
        print("img_temp/255.0")
        print(img_temp)

    # 自己写方法，实现灰度
    for i in range(h):
        for j in range(w):
            #取出当前high和wide中的BGR坐标
            m = img_temp[i,j]
            
            #将BGR坐标转化为gray坐标并赋值给新图像

            if(is_decimal):
                img_gray[i,j] = m[0]*0.11 + m[1]*0.59 + m[2]*0.3
            else:
                img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

    return img_gray

def img_gray_02(img):
    """
    灰度化，使用cv2内置函数
    结果还是整数
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def img_gray_03(img):
    """
    灰度化，使用skimage内置函数
    结果是0-1之间的浮点数
    """
    img_gray = rgb2gray(img)
    print(img_gray.dtype)
    return img_gray

def img_gray_04(img):
    """
    灰度化，自己写方法
    结果是0-1之间的浮点数
    """
    img_gray = img_gray_01(img,is_decimal=True)
    return img_gray

if __name__=="__main__":

    # 原图
    img_path="206-田海龙-北京/第02周/img/lenna.png"
    img=cv_imread(img_path)

    # BGRA转RGB
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(231)
    plt.imshow(img)

    # 再变回去
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_gray=img_gray_01(img)
    plt.subplot(232)
    plt.imshow(img_gray, cmap='gray')
    # print(img_gray)

    img_gray=img_gray_02(img)
    plt.subplot(233)
    plt.imshow(img_gray, cmap='gray')
    print(img_gray)

    img_gray=img_gray_03(img)
    plt.subplot(234)
    plt.imshow(img_gray, cmap='gray')
    print(img_gray)

    img_gray=img_gray_04(img)
    plt.subplot(235)
    plt.imshow(img_gray, cmap='gray')
    print(img_gray)

    plt.show()