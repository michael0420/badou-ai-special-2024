"""
作业：实现sobel边缘检测
"""
import cv2

def sobel_image():
    """
    图片边缘检测
    :return:
    """
    # cv2.IMREAD_GRAYSCALE：读入灰度图片，可用0作为实参替代
    img = cv2.imread("lenna.png",0)

    # 水平边缘检测， 原图为uint8 0-255,改为cv2.CV_16S：-32768-32767
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    # 垂直边缘检测
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    # 将Sobel水平边缘检测结果的负数值转换为正数
    absX = cv2.convertScaleAbs(x)
    # 将Sobel垂直边缘检测结果的负数值转换为正数
    absY = cv2.convertScaleAbs(y)
    # 使用addWeighted函数将水平和垂直边缘检测结果叠加，创建合并的边缘检测图像
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 显示水平检测图像
    cv2.imshow("absX", absX)
    # 显示垂直检测图像
    cv2.imshow("absY", absY)
    # 显示合并后显示图像
    cv2.imshow("result", dst)
    cv2.waitKey(0)


if __name__ == '__main__':
    sobel_image()