import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import bilinear


def function(img):#图片格式转为800*800
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800 / height
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            x = int(i / sh )  # int(),转为整型，使用向下取整。
            y = int(j / sw )
            emptyImage[i, j] = img[x, y]
    return emptyImage

def convert(img):
    # 灰度化
    h,w=img.shape[:2]
    # 1.创建新图
    emptyImage = np.zeros((h, w),img.dtype)#图的长宽，和数据类型需要确定一下
    # 2.逐个像素转化
    for i in range(h):
        for j in range(w):
            emptyImage[i,j]=int(img[i,j,0]*0.11+img[i,j,1]*0.59+img[i,j,2]*0.3)
    return emptyImage







def __preprocess_sampler_input(image, coords):
    """
    Nearest Neighbour sampler

    Parameters
    ----------
    image: ndarray
        source image, whose shape is [height, width, channel] or
        [height, width]
    coords: ndarray
        coordinates to be interpolated, the length of last axis should be 2,
        meaning 2D coordinate

    Returns
    -------
    image: ndarray
        source image. if its original shape is [height, width], it will be
        expanded with a new axis to have a shape of [height, width, 1]; if its
        original shape is [height, width, channel], it will not be changed
    coords: ndarray
        reshape from the original [n1, n2, ..., 2] to [n, 2],
        where n = n1 * n2 * ...
    output_shape: list
        the output shape of sampler function, same as coords expcept the last axis.
    """
    assert image.ndim == 2 or image.ndim == 3
    # cache output_shape
    output_shape = list(coords.shape)
    if image.ndim == 2:
        output_shape.pop()
        image = np.expand_dims(image, axis=-1)#添加最后一个维度
    else:
        output_shape[-1] = image.shape[-1]
    coords = np.reshape(coords, (-1, coords.shape[-1]))

    # 这段代码是使用
    # NumPy
    # 库的reshape函数将一个多维数组重新塑造成指定形状。其中的 - 1
    # 表示该位置的维度是由函数自动计算的。在这段代码中，reshape函数的参数为(-1, 1)，表示将原来的数组重新塑造为只有一列的二维数组。其中的 - 1
    # 代表该位置的维度将由函数根据数组元素的个数进行计算，以使新数组能够包含所有原始数组中的元素，而1代表新数组只有一列。
    return image, coords, output_shape
    #数据处理后image有3个维度，





def nearest(image, coords):
    """
    Nearest Neighbour sampler

    Parameters
    ----------
    image: ndarray
        source image, whose shape is [height, width, channel] or
        [height, width]
    coords: ndarray
        coordinates to be interpolated, the length of last axis should be 2,
        meaning 2D coordinate

    Returns
    -------
    output: ndarray
        the interpolated image, same shape as coords except the last axis
    """
    image, coords, output_shape = __preprocess_sampler_input(image, coords)
    height, width, channel = image.shape
    coords = np.round(coords).astype(np.int32)
    idx = (coords[:, 0] >= 0) & (coords[:, 0] < width) & \
          (coords[:, 1] >= 0) & (coords[:, 1] < height)
    output = np.zeros((coords.shape[0], channel), dtype=np.uint8)
    output[idx] = image[coords[idx, 1], coords[idx, 0]]
    # reshape back to the output_shape
    output = np.reshape(output, output_shape)
    return output

def double_nearest_native(image,coords):
    image, coords, output_shape = __preprocess_sampler_input(image, coords)
    height, width, channel = image.shape
    # coords = np.round(coords).astype(np.int32)
    x = 1
    y = 0
    output = np.zeros((coords.shape[0], channel), dtype=np.uint8)
    for i, coord in enumerate(coords):  # coords中含有变换后的坐标系
        if 0 <= coord[0] < width-1 and 0 <= coord[1] < height-1:

            fst_x1=int(coord[x])
            fst_x2=int(coord[x])+1
            fst_y1=int(coord[y])
            fst_y2=int(coord[y])+1
            lft1 = image[fst_x1, fst_y1]
            rig1 = image[fst_x2, fst_y1]
            up=lft1*(coord[x]-fst_x1)/(fst_x2-fst_x1)+rig1*(fst_x2-coord[x])/(fst_x2-fst_x1)

            lft2 = image[fst_x1, fst_y2]
            rig2 = image[fst_x2, fst_y2]
            down=lft2*(coord[x]-fst_x1)/(fst_x2-fst_x1)+rig2*(fst_x2-coord[x])/(fst_x2-fst_x1)

            output[i]=up*(coord[y]-fst_y1)/(fst_y2-fst_y1)+down*(fst_y2-coord[y])/(fst_y2-fst_y1)


            # output[i] = image[coord[1], coord[0]]  # i为索引（原坐标）
    # reshape back to the output_shape
    output = np.reshape(output, output_shape)
    return output

def nearest_naive(image, coords):
    image, coords, output_shape = __preprocess_sampler_input(image, coords)
    height, width, channel = image.shape
    coords = np.round(coords).astype(np.int32)
    output = np.zeros((coords.shape[0], channel), dtype=np.uint8)
    for i, coord in enumerate(coords):#coords中含有变换后的坐标系
        if 0 <= coord[0] < width and 0 <= coord[1] < height:
            output[i] = image[coord[1], coord[0]]#i为索引（原坐标）
    # reshape back to the output_shape
    output = np.reshape(output, output_shape)
    return output

# coordinate transform
# T = np.array([[1.7, -0.5],
#               [0.5, 1.7],
#               [0, 0]])

T = np.array([[1.2, 0.],
              [0., 1.2]])
print(type(T))
image = cv2.imread('lenna.png', cv2.IMREAD_UNCHANGED)
height, width = image.shape[0:2]
coords = np.meshgrid(np.arange(0, width), np.arange(0, height))
print(type(coords))
# shape = [height, width, 2] after transpose
coords = np.array(coords)
print(type(coords))

coords=coords.transpose([1, 2, 0])
ones = np.ones([height, width, 1])
# homogeneous  coordinates
# coords = np.concatenate((coords, ones), axis=2)
# transformed coordinates
coords = coords @ T#三维的乘法？
#coords是经过处理后的三维
# sampler
t0 = time.time()
cv2.imshow("t1",image)
img = double_nearest_native(image, coords)  # try different functions here
img2=nearest_naive(image,coords)
cv2.imshow("title",img)
cv2.waitKey(0)
cv2.imshow("title2",img2)
t_cost = time.time() - t0
print(t_cost)
cv2.waitKey(0)


