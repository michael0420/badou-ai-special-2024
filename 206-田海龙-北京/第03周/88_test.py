
import cv2

def read_img(img_path):
    img = cv2.imread(img_path)
    return img

def show_img(img,title):
    cv2.imshow(title,img)
    cv2.waitKey()

def draw_img(img,content):
    cv2.putText(img,content,(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    show_img(img,'测试-2')

def img_test():
    img=read_img(r'D:\Temp\测试\lenna.png')
    print(img) # the result is None

    img=read_img(r'D:\Temp\Test\lenna.png')
    show_img(img,'测试')

    draw_img(img,'测试-1')

import cv2
import numpy as np


# 加载RGB图像
img_rgb = cv2.imread(r"D:\Desktop\cat and rabbit.jpg")

# 检查图像是否正确加载
if img_rgb is not None:

    # 将图像从BGR（OpenCV默认）转换为RGB
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    # 将图像转为float类型并归一化到[0, 1]区间，便于进行直方图均衡化
    img_float = img_rgb.astype(np.float32) / 255.0

    # 分别对RGB三个通道进行直方图均衡化
    red_channel = img_float[..., 0]  # 取出红色通道
    red_channel_uint8 = np.clip(red_channel * 255, 0, 255).astype(np.uint8)  # 归一化后乘以255并转换为uint8
    red_eq = cv2.equalizeHist(red_channel_uint8.reshape(-1, 1)).reshape(img_rgb.shape[:2])
    # red_eq = cv2.equalizeHist(img_float[..., 0].reshape(-1, 1)).reshape(img_rgb.shape[:2])

    green_channel = img_float[..., 1]  # 取出绿色通道
    green_channel_uint8 = np.clip(green_channel * 255, 0, 255).astype(np.uint8)  # 归一化后乘以255并转换为uint8
    green_eq = cv2.equalizeHist(green_channel_uint8.reshape(-1, 1)).reshape(img_rgb.shape[:2])
    # green_eq = cv2.equalizeHist(img_float[..., 1].reshape(-1, 1)).reshape(img_rgb.shape[:2])

    blue_channel = img_float[..., 2]  # 取出蓝色通道
    blue_channel_uint8 = np.clip(blue_channel * 255, 0, 255).astype(np.uint8)  # 归一化后乘以255并转换为uint8
    blue_eq = cv2.equalizeHist(blue_channel_uint8.reshape(-1, 1)).reshape(img_rgb.shape[:2])
    # blue_eq = cv2.equalizeHist(img_float[..., 2].reshape(-1, 1)).reshape(img_rgb.shape[:2])

    # 重新构建均衡化后的RGB图像
    eq_rgb = np.stack((red_eq, green_eq, blue_eq), axis=-1)

    # 将结果转换回uint8类型，并从[0, 1]区间重新映射到[0, 255]区间
    eq_rgb = (eq_rgb ).astype(np.uint8)

    # 显示原图和均衡化后的图像
    img_rgb=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)
    cv2.imshow('Original Image', img_rgb)
    eq_rgb=cv2.cvtColor(eq_rgb,cv2.COLOR_RGB2BGR)
    cv2.imshow('Histogram Equalized Image', eq_rgb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Failed to load the image.")