import cv2
import numpy as np
from scipy.signal import convolve2d
from P1anpian_Convolve2d import Convolve2d_Diy

'''DIY实现Sobel边缘检测'''
def Sobel_Diy(img):
    # Sobel算子定义
    # 检测水平边缘
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    # 检测垂直边缘
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1,-2,-1]])

    # 使用库函数
    print(img,sobel_x)
    sobel_x_image = convolve2d(img, sobel_x, mode='same')
    sobel_y_image = convolve2d(img, sobel_y, mode='same')
    gradient_magnitude = np.sqrt(sobel_x_image ** 2 + sobel_y_image ** 2) # 梯度幅度图像
    abs_sobel_x_image = cv2.convertScaleAbs(sobel_x_image)
    abs_sobel_y_image = cv2.convertScaleAbs(sobel_y_image)
    dst_sobel1 = cv2.addWeighted(abs_sobel_x_image, 0.5, abs_sobel_y_image, 0.5, 0)

    # 使用diy函数
    sobel_x_image_Diy = Convolve2d_Diy(img, sobel_x)
    sobel_y_image_Diy = Convolve2d_Diy(img, sobel_y)
    abs_sobel_x_image_Diy = cv2.convertScaleAbs(sobel_x_image_Diy)
    abs_sobel_y_image_Diy = cv2.convertScaleAbs(sobel_y_image_Diy)
    dst_sobel2 = cv2.addWeighted(abs_sobel_x_image_Diy, 0.5, abs_sobel_y_image_Diy, 0.5, 0)

    return dst_sobel1, dst_sobel2

# 测试
img = cv2.imread("lenna.png", 0) # 读入图片 并转换为灰度图像
# dst_sobel1 2都是diy函数的结果
dst_sobel1, dst_sobel2 = Sobel_Diy(img)
cv2.imshow("sobel1", dst_sobel1)
cv2.imshow("sobel2", dst_sobel2)

'''使用CV2库实现Sobel检测'''
'''
Sobel函数求完导数后会有负值，还有会大于255的值。
而原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。
因此要使用16位有符号的数据类型，即cv2.CV_16S。
'''
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

'''
在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
否则将无法显示图像，而只是一副灰色的窗口。
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
'''

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

'''
由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
。其函数原型为：
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。
'''

# 测试2
dst3 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX3", absX)
cv2.imshow("absY3", absY)
cv2.imshow("Result3", dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()
