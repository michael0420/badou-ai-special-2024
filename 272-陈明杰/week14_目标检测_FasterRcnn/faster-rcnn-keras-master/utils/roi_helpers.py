import numpy as np
import pdb
import math
import copy
import time

def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h

def iou(a, b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)

# R是经过初步筛选的预测框,all_boxes是图片上所有正确答案的框
def calc_iou(R, config, all_boxes, width, height, num_classes):
    # print(all_boxes)
    bboxes = all_boxes[:,:4]
    gta = np.zeros((len(bboxes), 4))
    for bbox_num, bbox in enumerate(bboxes):
        gta[bbox_num, 0] = int(round(bbox[0]*width/config.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox[1]*height/config.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox[2]*width/config.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox[3]*height/config.rpn_stride))
    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []
    # print(gta)
    for ix in range(R.shape[0]):
        x1 = R[ix, 0]*width/config.rpn_stride
        y1 = R[ix, 1]*height/config.rpn_stride
        x2 = R[ix, 2]*width/config.rpn_stride
        y2 = R[ix, 3]*height/config.rpn_stride

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        # print([x1, y1, x2, y2])
        best_iou = 0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num
        # print(best_iou)
        if best_iou < config.classifier_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if config.classifier_min_overlap <= best_iou < config.classifier_max_overlap:
                label = -1
            elif config.classifier_max_overlap <= best_iou:

                label = int(all_boxes[best_bbox,-1])
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 2]) / 2.0
                cyg = (gta[best_bbox, 1] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 2] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 1]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError
        # print(label)
        class_label = num_classes * [0]
        class_label[label] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (num_classes - 1)
        labels = [0] * 4 * (num_classes - 1)
        if label != -1:
            label_pos = 4 * label
            sx, sy, sw, sh = config.classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    X = np.array(x_roi)
    # print(X)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs


def calc_iou(R, config, all_boxes, width, height, num_classes):
    # all_boxes 是一个二维数组，其中每一行代表一个边界框，通常包含四个坐标：[x_min, y_min, x_max, y_max]
    # 这些坐标是相对于原始图像尺寸的

    # 提取 all_boxes 中的边界框坐标，即每一行的前四个元素
    bboxes = all_boxes[:, :4]

    # 初始化一个与 bboxes 形状相同的数组 gta，用于存储转换后的坐标
    # 这里的 4 表示边界框的四个坐标（x_min, y_min, x_max, y_max）
    gta = np.zeros((len(bboxes), 4))

    # 遍历每一个边界框
    for bbox_num, bbox in enumerate(bboxes):
        # bbox 是一个包含四个坐标的数组：[x_min, y_min, x_max, y_max]
        # 将边界框的坐标从原始图像空间转换到与 config.rpn_stride 相对应的空间
        # 这通常是因为 RPN 或后续处理步骤在特征图上进行，特征图的尺寸是原始图像尺寸的 1/stride

        # 计算并存储转换后的 x_min 坐标
        gta[bbox_num, 0] = int(round(bbox[0] * width / config.rpn_stride))
        # 计算并存储转换后的 y_min 坐标
        gta[bbox_num, 1] = int(round(bbox[1] * height / config.rpn_stride))
        # 计算并存储转换后的 x_max 坐标
        gta[bbox_num, 2] = int(round(bbox[2] * width / config.rpn_stride))
        # 计算并存储转换后的 y_max 坐标
        gta[bbox_num, 3] = int(round(bbox[3] * height / config.rpn_stride))


    # 初始化变量来存储结果
    x_roi = []  # 用于存储提议区域的坐标和尺寸（x1, y1, width, height）
    y_class_num = []  # 在代码片段中未直接使用，但通常用于存储分类标签
    y_class_regr_coords = []  # 在代码片段中未直接使用，但通常用于存储边界框回归的坐标目标
    y_class_regr_label = []  # 在代码片段中未直接使用，但通常用于存储边界框回归的标签（可能区分正样本和负样本）
    IoUs = []  # 用于存储每个提议区域与其最佳匹配真实边界框的交并比（IOU）

    # 遍历RPN输出的每个提议区域
    for ix in range(R.shape[0]):
        # 计算提议区域的坐标（从相对于步长的尺寸转换为相对于原始图像尺寸的整数坐标）
        x1 = R[ix, 0] * width / config.rpn_stride
        y1 = R[ix, 1] * height / config.rpn_stride
        x2 = R[ix, 2] * width / config.rpn_stride
        y2 = R[ix, 3] * height / config.rpn_stride

        # 将浮点数坐标四舍五入并转换为整数
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        # 初始化最佳IOU和最佳匹配的真实边界框索引
        best_iou = 0.0
        best_bbox = -1

        # 遍历所有真实边界框，找到与当前提议区域IOU最高的真实边界框
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        # 检查最佳IOU是否满足最小重叠阈值
        if best_iou < config.classifier_min_overlap:
            continue  # 如果不满足，则跳过当前提议区域
        else:
            # 计算提议区域的宽度和高度
            w = x2 - x1
            h = y2 - y1

            # 将提议区域的坐标和尺寸添加到x_roi列表中
            x_roi.append([x1, y1, w, h])

            # 将最佳IOU添加到IoUs列表中
            IoUs.append(best_iou)

            # 根据IOU的值确定分类标签和边界框回归目标（此部分在代码片段中未完全展示）
            if config.classifier_min_overlap <= best_iou < config.classifier_max_overlap:
                # 如果IOU在最小和最大重叠阈值之间，通常设置为背景（这里假设标签为-1）
                label = -1  # 注意：这里实际上没有将label添加到任何列表中
            elif config.classifier_max_overlap <= best_iou:
                # 如果IOU大于或等于最大重叠阈值，则计算边界框回归目标
                label = int(all_boxes[best_bbox, -1])  # 假设all_boxes的最后一列是类别标签

                # 计算中心坐标和宽度高度的偏移量（边界框回归目标）
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 2]) / 2.0
                cyg = (gta[best_bbox, 1] + gta[best_bbox, 3]) / 2.0
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 2] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 1]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError
        # 假设这是在一个函数内部，且已经定义了变量num_classes, config, x_roi, IoUs, 以及相关的边界框信息

        # 假设label是从某个地方获取的当前提议区域的真实类别标签
        # print(label)  # 这行代码被注释掉了，如果取消注释，它会打印出label的值

        # 创建一个长度为num_classes的全零列表，用于独热编码
        class_label = num_classes * [0]

        # 将真实类别label对应的索引位置设为1，其余位置保持为0
        class_label[label] = 1

        # 将独热编码添加到y_class_num列表中，y_class_num用于存储所有提议区域的分类标签
        y_class_num.append(copy.deepcopy(class_label))  # 假设copy已经被导入

        # 初始化一个列表，用于存储边界框回归的坐标，长度为4 * (num_classes - 1)，因为背景类通常不设置回归目标
        coords = [0] * 4 * (num_classes - 1)

        # 初始化一个列表，用于存储边界框回归的标签，同样长度为4 * (num_classes - 1)
        labels = [0] * 4 * (num_classes - 1)

        # 如果label不是背景（即label != -1），则计算边界框回归目标
        if label != -1:
            # 计算当前类别对应的坐标起始位置
            label_pos = 4 * label

            # 从配置中获取边界框回归的标准差（可能是用于缩放回归目标）
            sx, sy, sw, sh = config.classifier_regr_std

            # 计算缩放后的边界框回归目标
            coords[label_pos:4 + label_pos] = [sx * tx, sy * ty, sw * tw, sh * th]  # 这里假设tx, ty, tw, th已经被计算

            # 设置回归标签为全1（表示这些坐标是有效的回归目标）
            labels[label_pos:4 + label_pos] = [1, 1, 1, 1]

            # 将计算出的回归坐标和标签添加到相应的列表中
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            # 如果label是背景，则仍然将初始化的coords和labels添加到列表中
            # 注意：这在实际应用中可能不是必需的，因为背景通常不参与回归训练
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    # 检查是否有有效的提议区域
    if len(x_roi) == 0:
        # 如果没有提议区域，则返回四个None值
        return None, None, None, None

    # 准备训练数据
    X = np.array(x_roi)  # 将提议区域列表转换为NumPy数组
    Y1 = np.array(y_class_num)  # 将分类标签列表转换为NumPy数组

    # 将回归标签和回归坐标合并为一个二维数组
    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    # 使用np.expand_dims在每个数组的第一个维度上增加一维
    # 这通常是为了满足深度学习模型的输入要求，即期望的批处理维度
    X_expanded = np.expand_dims(X, axis=0)
    Y1_expanded = np.expand_dims(Y1, axis=0)
    Y2_expanded = np.expand_dims(Y2, axis=0)

    # 返回处理后的数据
    return X_expanded, Y1_expanded, Y2_expanded, IoUs