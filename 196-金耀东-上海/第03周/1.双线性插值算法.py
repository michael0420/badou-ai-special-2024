import numpy as np
import cv2
import os

# 使用最临近插值算法计算像素值
def _interp_nearest(src_img, x, y, c):
    return src_img[round(x), round(y), c]

# 使用双线性插值算法计算像素值
def _interp_2d(src_img, x, y, channel):
    """

        P(x0 , y0)-------------------------P(x1 , y0)
            |              |                   |
            |      S00     |       S10         |
            |              |                   |
            |----------P(x , y)----------------|
            |              |                   |
            |              |                   |
            |      S01     |       S11         |
            |              |                   |
            |              |                   |
        P(x0 , y1)-------------------------P(x1 , y1)

    点(x , y)的像素值为临近4个点的像素值与其权重的乘积再求和，每个点的权重值即为与该点相对角的矩阵的面积，因此：
    P(x , y) = S11 * P(x0 , y0) + S01 * P(x1 , y0) + S10 * P(x0 , y1) + S00 * P(x1 , y1)
    """

    # 虚拟点落在原图像的最边缘像素上时，使用最临近插值算法计算像素值
    if x<0 or y < 0 or x > src_img.shape[0] or y > src_img.shape[1]:
        return _interp_nearest(src_img, x, y, channel)

    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, src_img.shape[0]-1) , min(y0 + 1, src_img.shape[1]-1)
    u , v = x - x0 , y - y0
    return (
            (1 - u) * (1 - v) * src_img[x0, y0, channel] +
            (  u  ) * (1 - v) * src_img[x1, y0, channel] +
            (1 - u) * (  v  ) * src_img[x0, y1, channel] +
            (  u  ) * (  v  ) * src_img[x1, y1, channel]
    )

# 缩放图像
def _resize_img(src_img, dst_height, dst_width, interp_func):
    src_height, src_width, channels = src_img.shape
    dst_img = np.zeros([dst_height, dst_width, channels], dtype=float)
    normalized_src_img = to_normalize(src_img)
    sh, sw = dst_height / src_height, dst_width / src_width

    for h in range(dst_height):
        for w in range(dst_width):
            for c in range(channels):
                dst_img[h, w, c] = interp_func(normalized_src_img, get_index(h, sh), get_index(w, sw), c)

    return inverse_normalize(dst_img)

# 计算虚拟点的下标
def get_index(index, scale_ratio):
    return (index + 0.5) / scale_ratio - 0.5

# 图像矩阵归一化
def to_normalize(img, max_value=255.0):
    return img / max_value


# 图像矩阵反归一化
def inverse_normalize(img, max_value=255.0):
    return (img * max_value).astype(np.uint8)


# 使用最邻近插值算法缩放图像
def resize_interp_nearest(src_img, dst_height, dst_width):
    return _resize_img(src_img, dst_height, dst_width, interp_func=_interp_nearest)


# 使用双线性插值算法缩放图像
def resize_interp_2d(src_img, dst_height, dst_width):
    return _resize_img(src_img, dst_height, dst_width, interp_func=_interp_2d)

if __name__ == "__main__":
    # 获取图像文件路径
    base_dir_path = "img"
    img_filename = "lenna.jpg"
    img_path = os.path.join(base_dir_path, img_filename)

    # 缩放后图像的尺寸
    dst_height, dst_width = 800, 800

    # 读取源图像
    img = cv2.imread(img_path)

    # 使用最临近插值算法缩放图像
    img_interp_nr = resize_interp_nearest(img, dst_height, dst_width)

    # 使用双线性插值算法缩放图像
    img_interp_2d = resize_interp_2d(img, dst_height, dst_width)

    # 展示图像
    imgs = [img, img_interp_nr, img_interp_2d]
    titles = ["original", "interp_nr", "interp_2d"]
    for img, title in zip(imgs, titles):
        cv2.imshow(title, img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


