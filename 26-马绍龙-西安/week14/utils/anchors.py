import numpy as np
import keras
from utils.config import Config

# 初始化配置
config = Config()


def generate_anchors(sizes=None, ratios=None):
    """
    生成预定义的锚点（anchors）。

    锚点是物体检测中的一个重要概念，它们是滑动窗口在不同尺度和长宽比下的候选框。

    参数:
    sizes: 一个列表，包含锚点的宽度和高度的规模。
    ratios: 一个列表，包含锚点的长宽比。

    返回:
    一个numpy数组，每行代表一个锚点，每列分别是锚点的左上角x，左上角y，右下角x，右下角y。
    """
    if sizes is None:
        sizes = config.anchor_box_scales

    if ratios is None:
        ratios = config.anchor_box_ratios

    num_anchors = len(sizes) * len(ratios)

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T

    for i in range(len(ratios)):
        anchors[3 * i:3 * i + 3, 2] = anchors[3 * i:3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i:3 * i + 3, 3] = anchors[3 * i:3 * i + 3, 3] * ratios[i][1]

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def shift(shape, anchors, stride=config.rpn_stride):
    """
    为每个位置生成偏移量，将锚点从网格中心移动到每个像素位置。

    参数:
    shape: 输入图像的高度和宽度。
    anchors: 生成的锚点。
    stride: 锚点的步长。

    返回:
    偏移后的锚点。
    """
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=0)

    shifts = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]

    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]),
                                                                                keras.backend.floatx())
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    return shifted_anchors


def get_anchors(shape, width, height):
    """
    根据输入图像的尺寸，生成相对于输入图像的锚点。

    参数:
    shape: 输入图像的高度和宽度。
    width: 输入图像的实际宽度。
    height: 输入图像的实际高度。

    返回:
    相对于输入图像尺寸的锚点。
    """
    anchors = generate_anchors()
    network_anchors = shift(shape, anchors)
    network_anchors[:, 0] = network_anchors[:, 0] / width
    network_anchors[:, 1] = network_anchors[:, 1] / height
    network_anchors[:, 2] = network_anchors[:, 2] / width
    network_anchors[:, 3] = network_anchors[:, 3] / height
    network_anchors = np.clip(network_anchors, 0, 1)
    return network_anchors
