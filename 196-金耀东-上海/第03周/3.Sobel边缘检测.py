import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# 灰度级范围0-255
MIN_GRAY_LEVEL , MAX_GRAY_LEVEL = 0 , 255

# 边缘填充像素0
def add_padding(src_img, padding_size):
    src_height, src_width = src_img.shape[:2]
    dst_height, dst_width = src_height + 2 * padding_size, src_width + 2 * padding_size
    dst_img = np.zeros(shape=(dst_height, dst_width), dtype=np.uint8)
    for h in range(src_height):
        for w in range(src_width):
            dst_img[h + padding_size, w + padding_size] = src_img[h, w]
    return dst_img

# 卷积函数
def convolve2d(img, kernel, step=1):
    kernel_size = kernel.shape[0]

    # 边缘填充
    padding_img = add_padding( img, padding_size=int(kernel_size / 2) )

    # 目标图像尺寸
    dst_height = int( (padding_img.shape[0] - kernel_size + 1) / step )
    dst_width = int( (padding_img.shape[1] - kernel_size + 1) / step )

    dst_img = np.zeros(shape=(dst_height, dst_width))

    # 最大像素值与最小像素值
    min_val, max_val = MIN_GRAY_LEVEL , MAX_GRAY_LEVEL
    for h in range(dst_height):
        for w in range(dst_width):
            # 截取源图中的子块进行卷积运算，并将计算值填充到新图像中
            tmp_mat = padding_img[h*step : h*step+kernel_size , w*step : w*step+kernel_size]
            dst_img[h, w] =  (tmp_mat * kernel ).sum()

            # 更新最大像素值与最小像素值
            min_val = min(dst_img[h, w], min_val)
            max_val = max(dst_img[h, w], max_val)

    # 将计算结果的灰度级映射到0-255的范围内
    dst_img =  (dst_img - min_val) / (max_val - min_val) * MAX_GRAY_LEVEL
    return dst_img.astype("uint8")

# 展示图像
def display_imgs(imgs, titles, types, rows, cols):
    for i in range( len(imgs) ):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i], cmap=types[i])
        plt.title(titles[i])
        plt.xticks([]) , plt.yticks([]) # 不显示横纵坐标轴
    plt.show()

if __name__ == "__main__":
    img_dir = "img"
    img_filename = "lenna.jpg"
    img_path = os.path.join( img_dir, img_filename )
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 创建sobel卷积核
    soble_x = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
    soble_y = np.array(
        [
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])

    # 分别做x轴方向和y轴方向的边缘检测
    sobel_x_img = convolve2d( gray_img, soble_x )
    sobel_y_img = convolve2d( gray_img, soble_y )

    # 等权合并计算结果
    dst_img = (sobel_x_img*0.5 + sobel_y_img*0.5).astype("uint8")

    # 展示图片
    display_imgs(
        imgs=[gray_img, dst_img],
        titles=["gray image", "sobel image"],
        types=["gray", "gray"],
        rows=1, cols=2
    )
