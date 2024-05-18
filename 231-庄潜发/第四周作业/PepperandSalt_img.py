"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/4/20 10:48
"""
import random
import cv2


def get_gauss_img(percentage):
    # 生成存放噪声图像
    salt_img = gray_img
    # 计算图片总像素点
    pixel_sum = gray_img.shape[0]*gray_img.shape[1]
    # 计算你想要百分比的高斯噪声的像素点
    percents = int(pixel_sum * percentage)
    for i in range(percents):
        # 随机取原图中的行X和列Y, 边缘不处理
        randX = random.randint(0, gray_img.shape[0]-1)
        randY = random.randint(0, gray_img.shape[1]-1)
        # 随机得到像素值进行赋值
        if random.random() > 0.5:
            salt_img[randX, randY] = 255
        else:
            salt_img[randX, randY] = 0
    return salt_img


if __name__ == '__main__':
    # 参数0,表示图片读取为灰度图
    gray_img = cv2.imread("../lenna.png", 0)
    cv2.imshow("gray_img", gray_img)
    # 参数1:百分比噪声
    salt_img = get_gauss_img(0.3)
    cv2.imshow("salt_img", salt_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
