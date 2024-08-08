import sys
from operator import itemgetter
import numpy as np
import cv2
import matplotlib.pyplot as plt


# -----------------------------#
#   计算原始输入图像
#   每一次缩放的比例
# -----------------------------#
def calculateScales(img):
    # copy_img = img.copy()
    copy_img = img.copy()
    # pr_scale = 1.0
    pr_scale = 1.0
    # h, w, _ = copy_img.shape
    h, w, _ = copy_img.shape
    # 引申优化项  = resize(h*500/min(h,w), w*500/min(h,w))
    # if min(w, h) > 500:
    #     pr_scale = 500.0 / min(h, w)
    #     w = int(w * pr_scale)
    #     h = int(h * pr_scale)
    # elif max(w, h) < 500:
    #     pr_scale = 500.0 / max(h, w)
    #     w = int(w * pr_scale)
    #     h = int(h * pr_scale)
    if min(w, h) > 500:
        pr_scale = 500.0 / min(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)
    elif max(w, h) < 500:
        pr_scale = 500.0 / max(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)

    # scales = []
    # factor = 0.709
    # factor_count = 0
    # minl = min(h, w)
    # while minl >= 12:
    #     scales.append(pr_scale * pow(factor, factor_count))
    #     minl *= factor
    #     factor_count += 1
    # return scales
    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h, w)
    while minl > 12:
        scales.append(pr_scale * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales


# img=cv2.imread('./img/test1.jpg')
# x=calculateScales(img)
# print(x)


# -------------------------------------#
#   对pnet处理后的结果进行处理
# -------------------------------------#
# 这段代码的核心功能是处理由P-Net（或其他类似网络）生成的候选人脸区域，包括：
# 1、从分类概率图中筛选出高概率的候选区域。
# 2、使用候选区域的偏移量调整边界框的位置。
# 3、将边界框转换为更接近正方形的形状（尽管具体的转换方法未在代码中给出）。
# 4、筛选出在图像范围内的有效边界框。
# 5、对边界框执行非极大值抑制，消除重叠度过高的框，返回最终的边界框列表。
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    # 定义一个函数detect_face_12net，它接收多个参数：
    # cls_prob：分类概率图，通常是网络输出的每个像素点为人脸的概率
    # roi：候选区域偏移量，表示预测框相对于锚点的偏移
    # out_side：输出特征图的边长，用于计算步长
    # scale：当前图像金字塔层级的缩放比例
    # width, height：原始图像的宽度和高度
    # threshold：阈值，用于筛选高概率的候选区域

    # 交换cls_prob数组的轴，使其从(height, width)变为(width, height)
    # 通常这样的操作是为了方便后续按类别索引概率图
    # cls_prob = np.swapaxes(cls_prob, 0, 1)
    cls_prob = np.swapaxes(cls_prob, 0, 1)

    # 交换roi数组的轴，使其从(height, width,4)变为(4, width, height)
    # 这里假设roi包含了多个候选框的偏移量，每个候选框有4个偏移值（dx1, dy1, dx2, dy2）
    # roi = np.swapaxes(roi, 0, 2)
    roi = np.swapaxes(roi, 0, 2)

    # 初始化步长为0
    stride = 0

    # stride略等于2
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)
        # 如果out_side不等于1（即特征图不是1x1），则计算步长。这个步长用于将特征图上的坐标映射到原始图像坐标

    # (x, y) = np.where(cls_prob >= threshold)
    (x, y) = np.where(cls_prob >= threshold)
    # 找到cls_prob中大于或等于阈值的元素的索引，即高概率的候选区域位置

    # boundingbox = np.array([x, y]).T
    boundingbox = np.array([x, y]).T
    # 将找到的索引转换为边界框的初始坐标（只是特征图上的坐标）

    # 找到对应原图的位置
    # 使用步长和缩放比例将特征图上的坐标映射到原始图像坐标，并生成边界框的两个角点坐标
    # 注意这里的+11可能是为了获取一个稍微大一点的区域，但这取决于具体的网络设计和需求
    # bb1 = np.fix((stride * (boundingbox) + 0) * scale)
    # bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    bb1 = np.fix((stride * (boundingbox) + 0) * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)

    # plt.scatter(bb1[:,0],bb1[:,1],linewidths=1) 和以下两行是注释掉的代码，用于可视化边界框

    # 将两个角点的坐标合并为一个边界框坐标数组，每个边界框有4个坐标值(x1, y1, x2, y2)
    # boundingbox = np.concatenate((bb1, bb2), axis=1)
    boundingbox = np.concatenate((bb1, bb2), axis=1)

    # 从roi数组中提取对应高概率候选区域的偏移量
    # dx1 = roi[0][x, y]
    # dx2 = roi[1][x, y]
    # dx3 = roi[2][x, y]
    # dx4 = roi[3][x, y]
    dx1 = roi[0][x, y]
    dx2 = roi[1][x, y]
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]

    # 提取高概率候选区域的得分（即为人脸的概率）
    # score = np.array([cls_prob[x, y]]).T
    score = np.array([cls_prob[x, y]]).T

    # 将偏移量组合成一个数组
    offset = np.array([dx1, dx2, dx3, dx4]).T

    # 使用偏移量和缩放比例调整边界框的位置，这里乘以12.0可能是根据网络设计或经验确定的系数
    # boundingbox = boundingbox + offset * 12.0 * scale
    boundingbox = boundingbox + offset * 12.0 * scale

    # 将边界框坐标和得分合并为一个数组，每个元素是一个边界框及其得分
    # rectangles = np.concatenate((boundingbox, score), axis=1)
    rectangles = np.concatenate((boundingbox, score), axis=1)

    # 调用rect2square函数将边界框转换为更接近正方形的形状
    rectangles = rect2square(rectangles)

    # 初始化一个空列表，用于存储筛选后的边界框
    pick = []

    for i in range(len(rectangles)):
        # 对于每个边界框，确保其坐标在图像范围内，并获取其得分
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]

        if x2 > x1 and y2 > y1:
            # 如果边界框的宽度和高度都大于0（即是一个有效的框），则将其添加到pick列表中
            pick.append([x1, y1, x2, y2, sc])

    # 调用NMS函数执行非极大值抑制，以消除重叠度过高的边界框，并返回处理后的边界框列表
    # 这里的0.3是非极大值抑制的阈值，表示两个边界框的交并比（IOU）大于该值时，会被视为重叠度过高
    return NMS(pick, 0.3)


# -----------------------------#
#   将长方形调整为正方形
# -----------------------------#
def rect2square(rectangles):
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    l = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
    return rectangles


# -------------------------------------#
#   非极大抑制
# -------------------------------------#
def NMS(rectangles, threshold):
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())
    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])  # I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


# -------------------------------------#
#   对Rnet处理后的结果进行处理
# -------------------------------------#
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2 - x1
    h = y2 - y1

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, 0.3)


# -------------------------------------#
#   对onet处理后的结果进行处理
# -------------------------------------#
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2 - x1
    h = y2 - y1

    pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
    pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
    pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
    pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
    pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                axis=1)

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, rectangles[i][4],
                         rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9],
                         rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])
    return NMS(pick, 0.3)
