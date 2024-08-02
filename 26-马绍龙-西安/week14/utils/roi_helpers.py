import numpy as np
import copy


def union(au, bu, area_intersection):  # 计算两个矩形区域的并集面积
    """
    计算两个矩形的并集面积。

    参数:
    au -- 矩形A的四个顶点坐标
    bu -- 矩形B的四个顶点坐标
    area_intersection -- 矩形A和B的交集面积

    返回:
    area_union -- 矩形A和B的并集面积
    """
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):  # 计算两个矩形区域的交集面积
    """
    计算两个矩形的交集面积。

    参数:
    ai -- 矩形A的四个顶点坐标
    bi -- 矩形B的四个顶点坐标

    返回:
    intersection_area -- 矩形A和B的交集面积
    """
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    """
    计算两个矩形的IOU（Intersection over Union）。

    参数:
    a -- 矩形A的四个顶点坐标
    b -- 矩形B的四个顶点坐标

    返回:
    iou_value -- 矩形A和B的IOU值
    """
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def calc_iou(R, config, all_boxes, width, height, num_classes):
    """
    计算每个提案区域（R）与所有边界框（all_boxes）的IOU，并根据配置（config）进行筛选。

    参数:
    R -- 提案区域的矩形坐标
    config -- 配置参数，包含阈值等信息
    all_boxes -- 所有边界框的矩形坐标及类别标签
    width -- 图像宽度
    height -- 图像高度
    num_classes -- 类别的数量

    返回:
    x_roi -- 选择的提案区域坐标
    y_class_num -- 每个提案区域的类别计数
    y_class_regr_coords -- 每个提案区域的类别回归坐标
    y_class_regr_label -- 每个提案区域的类别回归标签
    """
    # 将边界框的坐标从分数转换为像素
    bboxes = all_boxes[:, :4]
    gta = np.zeros((len(bboxes), 4))
    for bbox_num, bbox in enumerate(bboxes):
        gta[bbox_num, 0] = int(round(bbox[0] * width / config.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox[1] * height / config.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox[2] * width / config.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox[3] * height / config.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []

    # 遍历每个提案区域，计算与每个边界框的IOU
    for ix in range(R.shape[0]):
        x1 = R[ix, 0] * width / config.rpn_stride
        y1 = R[ix, 1] * height / config.rpn_stride
        x2 = R[ix, 2] * width / config.rpn_stride
        y2 = R[ix, 3] * height / config.rpn_stride

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        # 寻找与提案区域IOU最大的边界框
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        # 如果IOU低于最小重叠阈值，则跳过
        if best_iou < config.classifier_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            # 根据IOU的大小确定标签和回归坐标
            if config.classifier_min_overlap <= best_iou < config.classifier_max_overlap:
                label = -1
            elif config.classifier_max_overlap <= best_iou:
                label = int(all_boxes[best_bbox, -1])
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 2]) / 2.0
                cyg = (gta[best_bbox, 1] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 2] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 1]) / float(h))
            else:
                raise RuntimeError('Invalid IOU value')

        # 根据标签更新类别计数和回归坐标
        class_label = num_classes * [0]
        class_label[label] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (num_classes - 1)
        labels = [0] * 4 * (num_classes - 1)
        if label != -1:
            label_pos = 4 * label
            sx, sy, sw, sh = config.classifier_regr_std
            coords[label_pos:4 + label_pos] = [sx * tx, sy * ty, sw * tw, sh * th]
            labels[label_pos:4 + label_pos] = [1, 1, 1, 1]
        y_class_regr_coords.append(copy.deepcopy(coords))
        y_class_regr_label.append(copy.deepcopy(labels))

    # 如果没有符合条件的提案区域，则返回空值
    if len(x_roi) == 0:
        return None, None, None, None

    # 将结果转换为合适的形状以供进一步处理
    X = np.array(x_roi)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs
