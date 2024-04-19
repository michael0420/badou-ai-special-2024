
import cv2
import numpy as np

def sobel_demo(image):
    # image是输入图像，即要计算Sobel导数的图像。
    # cv2.CV_32F是输出图像的深度，表示输出图像的每个像素值是32位浮点数。这样可以保留Sobel导数的正负符号和小数部分。
    # 1是x方向的导数的阶数，表示你要计算图像在x方向（即水平方向）的一阶导数。这可以检测垂直方向的边缘。
    # 0是y方向的导数的阶数，表示你不计算图像在y方向（即垂直方向）的导数。
    # 所以，grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    # 这行代码的意思是，计算图像image在x方向的Sobel导数，并将结果存储在grad_x中
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1) #同理
    # cv2.convertScaleAbs()函数将其转回原来的uint8形式。否则将有可能得到负值，显示不出来。
    # cv2.Sobel函数返回的是一个32位浮点数数组grad_x，这个数组包含了图像在x方向的Sobel导数。
    # 这个数组的值可能是正数也可能是负数，而且可能包含小数部分。
    # 然而，OpenCV的图像显示函数cv2.imshow需要一个无符号整数数组。
    # 所以，需要使用cv2.convertScaleAbs函数将grad_x转换为一个无符号整数数组gradx，然后再显示它。
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    # cv2.addWeighted()函数将两个图像进行加权融合。这里的加权融合是指将两个图像的像素值按照一定的权重相加。
    # 第一个参数gradx是第一个图像，即x方向的Sobel导数。
    # 第二个参数0.5是第一个图像的权重。
    # 第三个参数grady是第二个图像，即y方向的Sobel导数。
    # 第四个参数0.5是第二个图像的权重。
    # 第五个参数0是一个常数，表示将两个图像的像素值相加时再加上这个常数。
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv2.imshow("gradient_x", gradx)
    cv2.imshow("gradient_y", grady)
    cv2.imshow("gradient", gradxy)
    return gradxy

img = cv2.imread("lenna.png", 0)
sobel_demo(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
