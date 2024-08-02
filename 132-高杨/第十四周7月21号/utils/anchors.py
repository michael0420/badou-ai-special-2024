import numpy as np
import keras
import tensorflow as tf
from Badou.第十四周7月21号.utils.config import Config
import matplotlib.pyplot as plt

config = Config()

def generate_anchors(sizes=None, ratios=None):
    if sizes is None:
        sizes = config.anchor_box_scales

    if ratios is None:
        ratios = config.anchor_box_ratios

    num_anchors = len(sizes) * len(ratios)

    anchors = np.zeros((num_anchors, 4))

    #  对anchors 中所有h，w进行操作，例如原数组是【128，256 ，512】 复制两次 赋值给隔壁两列

    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T

    # 循环三次  取ratio 中的比例 例如 1：1 1：2 2：1
    for i in range(len(ratios)):
        anchors[3*i:3*i+3, 2] = anchors[3*i:3*i+3, 2]*ratios[i][0]
        anchors[3*i:3*i+3, 3] = anchors[3*i:3*i+3, 3]*ratios[i][1]
    

    # 选择 每一行 每一行是一个锚点 的0 2 4列


    # # 创建一个示例数组 下面是用法
    # anchors = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    #
    # # 使用切片选择偶数索引列
    # even_index_columns = anchors[:, 0::2]
    #
    # print(even_index_columns)  # 输出: [[1 3 9]
    # [5 7 11]]
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors

def shift(shape, anchors, stride=config.rpn_stride):
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    # 获取经过卷积池化后的 图片大小 M*N ， shape一共有多少框
    k = np.shape(shifts)[0]

    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    return shifted_anchors

def get_anchors(shape,width,height):
    anchors = generate_anchors()
    network_anchors = shift(shape,anchors)
    network_anchors[:,0] = network_anchors[:,0]/width
    network_anchors[:,1] = network_anchors[:,1]/height
    network_anchors[:,2] = network_anchors[:,2]/width
    network_anchors[:,3] = network_anchors[:,3]/height
    network_anchors = np.clip(network_anchors,0,1)
    return network_anchors
