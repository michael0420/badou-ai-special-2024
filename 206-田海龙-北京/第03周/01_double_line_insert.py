import numpy as np
import cv2

import matplotlib.pyplot as plt

from Util import cv_imread

import matplotlib as mpl

# 设置matplotlib默认字体为支持中文的字体
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体（或其他支持中文的字体）
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方框的问题


img_path = "206-田海龙-北京/第03周/img/lenna.png"


def double_line_insert(img, out_shape):
    """
    双线性插值
    :param img: 输入图像
    :param out_shape: 输出图像的shape
    :return:
    """
    h, w, c = img.shape
    out_h, out_w = out_shape

    # 如果与原图像大小相同，直接返回结果
    if h == out_h and w == out_w:
        return img.copy()

    r_h = out_h / h
    r_w = out_w / w

    out_img = np.zeros((out_h, out_w, c), dtype=np.uint8)

    # 遍历通道，遍历新图像的像素点
    # 找出对应原图的点，进行计算
    for i in range(c):
        for j in range(out_h):
            for k in range(out_w):

                # 对应的原图坐标
                x = (k + 0.5) / r_w - 0.5
                y = (j + 0.5) / r_h - 0.5

                # 坐标点转为整数
                x0 = int(x)
                y0 = int(y)

                x1 = min(int(x0 + 1), w - 1)
                y1 = min(int(y0 + 1), h - 1)

                # 求解R1、R2点
                temp0 = (x1 - x) * img[y0, x0, i] + (x - x0) * img[y0, x1, i]
                temp1 = (x1 - x) * img[y1, x0, i] + (x - x0) * img[y1, x1, i]
                # 最终坐标点
                out_img[j, k, i] = int((y1 - y) * temp0 + (y - y0) * temp1)

    return out_img


is_print_time = True
import time


def print_now_time(info="start"):
    """
    打印时间，用于看下大概哪块花时间比较多
    """
    if is_print_time:
        time_str=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{time_str}------{info}")


print_now_time()
img = cv_imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print_now_time()
plt.subplot(131)
plt.imshow(img)
plt.title("原图")

print_now_time("准备开始双线性插值")
# 双线性插值
# 三层循环，速度有点慢
out_img = double_line_insert(img, (999, 999))
print_now_time("双线性插值完成")
plt.subplot(132)
plt.imshow(out_img)
plt.title("双线性插值")
plt.imsave(img_path.replace("img/lenna", "double_line_insert"), out_img)

print_now_time("准备开始cv2内部双线性插值")
# cv2内部双线性插值
out_img_02 = cv2.resize(img, (999, 999), interpolation=cv2.INTER_LINEAR)
print_now_time("cv2内部双线性插值完成")
plt.subplot(133)
plt.imshow(out_img_02)
plt.title("cv2内部双线性插值")
plt.imsave(img_path.replace("img/lenna", "double_line_insert_02"), out_img_02)

plt.show()
print_now_time()
