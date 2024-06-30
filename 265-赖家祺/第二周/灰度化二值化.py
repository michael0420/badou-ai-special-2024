import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

"""
cv2读取图片的通道顺序为BGR，与其他库不同
plt读取通道顺序为RGB

区别
目标不同：灰度化的目的是简化图像处理，去除颜色信息，保留亮度信息；而二值化的目的是提取图像的结构信息，将图像简化为黑白两色。
处理结果不同：灰度化后图像仍然是连续的灰度级别，而二值化后图像只有两种像素值。
应用场景不同：灰度化是很多图像处理任务的预处理步骤，而二值化更多用于需要突出图像特定结构信息的场景。
信息损失不同：灰度化相比于原始彩色图像，损失了颜色信息，但保留了亮度信息；二值化则进一步损失了灰度信息，只保留了最基础的结构信息。
"""


# 手动实现灰度化
def manual_gray():
    img = cv2.imread("./lenna.png")
    print("origin img first pixel: ", img[0, 0])  # [125 137 226]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"BGR TO RGB first pixel: {img_rgb[0, 0]}\n")  # [226 137 125]

    print("img.shape: ", img.shape)   # (512, 512, 3)
    height, width = img.shape[:2]
    print(f"img.dtype: {img.dtype, type(img.dtype)}\n")  # uint8 <class 'numpy.dtype[uint8]'>
    img_gray = np.zeros([height, width], img.dtype)
    cv2.imshow("my img gray before", img_gray)

    for h in range(height):
        for w in range(width):
            pixel = img[h, w]
            #                      B                 G               R
            img_gray[h, w] = int(pixel[0]*0.11 + pixel[1]*0.59 + pixel[2]*0.3)

    print("img_gray first pixel: ", img_gray[0, 0])
    print(f"img_gray shape: {img_gray.shape}\n")
    cv2.imshow("my img gray after", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 接口调用灰度化
def plt_gray():
    # 原图
    plt.subplot(321)
    img = plt.imread("./lenna.png")
    plt.imshow(img)
    print(img[0, :10])

    # 正常灰度化
    plt.subplot(322)
    img_gray = rgb2gray(img)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img_gray[0][:10])
    plt.imshow(img_gray, cmap='gray')

    # 手动二值化
    # rows, cols = img_gray.shape
    # for r in range(rows):
    #     for c in range(cols):
    #         if img_gray[r, c] < 0.5:
    #             img_gray[r, c] = 0
    #         else:
    #             img_gray[r, c] = 1
    # print(img_gray[0][:3])
    # img_binary = img_gray
    pass
    # 二值化
    plt.subplot(323)
    img_binary = np.where(img_gray >= 0.5, 1, 0)
    print(img_binary[0][:10])
    plt.imshow(img_binary, cmap='gray')

    # 错误灰度化
    plt.subplot(324)
    # plt已经读取为RGB, 这里故意按照BGR继续转换，生成的是错误的灰度图
    # 但肉眼观察生成的错误灰度图无明显差异
    img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img_gray2, cmap='gray')

    # 黑白转白黑
    plt.subplot(325)
    img_binary = np.where(img_gray >= 0.5, 0, 1)
    print("黑白转白黑: ", img_binary[0][:10])
    plt.imshow(img_binary, cmap='gray')

    # 彩虹色
    plt.subplot(326)
    img_binary = np.where(img_gray >= 0.5, 1, 0)
    plt.imshow(img_binary, cmap='rainbow')


if __name__ == '__main__':
    # manual_gray()
    plt_gray()

    plt.show()
    # plt.waitforbuttonpress()
