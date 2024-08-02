import os
import config
import random
import colorsys
import numpy as np
import tensorflow as tf
from model.yolo3_model import yolo

# 本文件为预测文件，用于单张图片的预测

"""
    加载模型，进行预测 
"""


class yolo_predictor:
    # 读取类别文件
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):  #
        self.obj_threshold = obj_threshold  # 目标检测为物体的阈值
        self.nms_threshold = nms_threshold  # nms阈值
        # 预读取
        self.classes_path = classes_file  # 类别文件
        self.anchors_path = anchors_file  # anchors文件
        # 读取种类名称
        self.class_names = self._get_class()  # 读取类别名称
        # 读取先验框
        self.anchors = self._get_anchors()  # 读取anchors数据

        # 画框框用
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]  # 生成颜色

        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))  # 转换颜色格式
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))  # 转换颜色格式
        random.seed(10101)  # 固定随机种子，使得每次生成的随机数相同，10101是随便取的
        random.shuffle(self.colors)  # 随机打乱颜色
        random.seed(None)  # 取消随机种子，使得每次生成的随机数都不同

    # 读取类别名称
    def _get_class(self):  # 读取类别名称
        classes_path = os.path.expanduser(self.classes_path)  # 读取类别文件
        with open(classes_path) as f:  # 打开类别文件
            class_names = f.readlines()  # 读取类别文件中的内容
        class_names = [c.strip() for c in class_names]  # 去除每一行的空格
        return class_names  # 返回类别名称

    # 读取anchors数据
    def _get_anchors(self):  # 读取anchors数据
        anchors_path = os.path.expanduser(self.anchors_path)  # 读取anchors文件
        with open(anchors_path) as f:  # 打开anchors文件
            anchors = f.readline()  # 读取anchors文件中的内容
            anchors = [float(x) for x in anchors.split(',')]  # 以逗号为分隔符，将anchors文件中的内容分割开来
            anchors = np.array(anchors).reshape(-1, 2)  # 将anchors数据转换为numpy数组
        return anchors  # 返回anchors数据

    # ---------------------------------------#
    #   对三个特征层解码
    #   进行排序并进行非极大抑制
    # ---------------------------------------#

    """
    对三个特征层解码，分别是13x13,26x26,52x52，将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数，进行排序并进行非极大抑制
    输入：
        feats: yolo输出的feature map
        anchors: anchor的位置
        classes_num: 类别数目
        input_shape: 输入大小
        image_shape: 图片大小
    输出：
        boxes: 物体框的位置
        boxes_scores: 物体框的分数，为置信度和类别概率的乘积
    """

    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        # 获得特征
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num,
                                                                          input_shape)  # 获得特征，分别是xy坐标，宽高，置信度，类别概率
        # 寻找在原图上的位置
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)  # 寻找在原图上的位置
        boxes = tf.reshape(boxes, [-1, 4])  # 转换为一维，方便后面进行计算
        # 获得置信度box_confidence * box_class_probs
        box_scores = box_confidence * box_class_probs  # 获得置信度box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])  # 转换为一维，方便后面进行计算
        return boxes, box_scores  # 返回物体框的位置和分数

    """
    获得在原图上框的位置
    """

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):  # 获得在原图上框的位置，输入的是xy坐标，宽高，输入大小，图片大小，输出的是物体框的位置
        box_yx = box_xy[..., ::-1]  # 物体框左上角坐标
        box_hw = box_wh[..., ::-1]  # 物体框的宽高
        # 416,416
        input_shape = tf.cast(input_shape, dtype=tf.float32)  # 输入的大小
        # 实际图片的大小
        image_shape = tf.cast(image_shape, dtype=tf.float32)  # 图片的大小

        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))  # 计算图片缩放后的大小

        offset = (input_shape - new_shape) / 2. / input_shape  # 计算图片缩放后的偏移量
        scale = input_shape / new_shape  # 计算图片缩放的比例
        box_yx = (box_yx - offset) * scale  # 计算物体框左上角坐标在原图中的位置坐标
        box_hw *= scale  # 计算物体框的宽高在原图中的位置坐标

        box_mins = box_yx - (box_hw / 2.)  # 计算物体框左上角坐标在原图中的位置坐标
        box_maxes = box_yx + (box_hw / 2.)  # 计算物体框右下角坐标在原图中的位置坐标
        boxes = tf.concat([  # 将左上角坐标和右下角坐标拼接起来
            box_mins[..., 0:1],  # 左上角x坐标，0：1表示取第0列到第1列
            box_mins[..., 1:2],  # 左上角y坐标，1：2表示取第1列到第2列
            box_maxes[..., 0:1],  # 右下角x坐标，0：1表示取第0列到第1列
            box_maxes[..., 1:2]  # 右下角y坐标，1：2表示取第1列到第2列
        ], axis=-1)  # 拼接，-1表示最后一维，即列
        boxes *= tf.concat([image_shape, image_shape], axis=-1)  # 将坐标转换为原图上的坐标，乘以原图的大小
        return boxes  # 返回物体框的位置

    """
    其实是解码的过程.根据yolo最后一层的输出确定bounding box
    输入的是yolo模型最后一层输出，anchors的位置，类别数量，输入大小，输出的是xy坐标，宽高，置信度，类别概率
    """

    def _get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchors = len(anchors)  # anchors的数量
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])  # 将anchors转换为张量，
        grid_size = tf.shape(feats)[1:3]  # 特征的大小，即13*13，1：3表示取第1维到第3维
        predictions = tf.reshape(feats,
                                 [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])  # 将特征转换为张量，方便后面进行计算

        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]),[1, grid_size[1], 1, 1])  # 构建13*13*1*1的矩阵，对应每个格子加上对应的坐标
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]),[grid_size[0], 1, 1, 1])  # 构建13*13*1*1的矩阵，对应每个格子加上对应的坐标
        grid = tf.concat([grid_x, grid_y], axis=-1)  # 将x,y坐标拼接起来
        grid = tf.cast(grid, tf.float32)  # 将坐标转换为浮点数

        # 将x,y坐标归一化，相对网格的位置
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)  # 将x,y坐标归一化，相对网格的位置
        # 将w,h也归一化
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1],
                                                                          tf.float32)  # 将w,h也归一化，predictions[..., 2:4]表示取第2列到第4列，archors_tensor表示anchors的大小，input_shape[::-1]表示输入的大小
        box_confidence = tf.sigmoid(predictions[..., 4:5])  # 计算置信度
        box_class_probs = tf.sigmoid(predictions[..., 5:])  # 计算类别概率
        return box_xy, box_wh, box_confidence, box_class_probs  # 返回物体框的位置，宽高，置信度，类别概率

    """
    根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
    输入: yolo模型输出，图片大小，最大box数量
    输出: 物体框的位置，物体类别的概率，物体类别
    """

    def eval(self, yolo_outputs, image_shape, max_boxes=20):  # 非极大值抑制，获取最后的物体检测框和物体检测类别
        # 每一个特征层对应三个先验框
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # 每一个特征层对应三个先验框
        boxes = []  # 物体框的位置
        box_scores = []  # 物体类别的概率
        # inputshape是416x416
        # image_shape是实际图片的大小
        input_shape = tf.shape(yolo_outputs[0])[
                      1: 3] * 32  # inputshape是416x416，image_shape是实际图片的大小，yolo_outputs[0]表示第一个特征层的输出，[1 : 3]表示取第1维到第3维，*32表示乘以32
        # 对三个特征层的输出获取每个预测box坐标和box的分数，score = 置信度x类别概率
        # ---------------------------------------#
        #   对三个特征层解码
        #   获得分数和框的位置
        # ---------------------------------------#
        for i in range(len(yolo_outputs)):  # 对三个特征层解码，获得分数和框的位置
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                        len(self.class_names), input_shape,
                                                        image_shape)  # 对三个特征层解码，获得分数和框的位置
            boxes.append(_boxes)  # 将每个特征层的box放在一起
            box_scores.append(_box_scores)  # 将每个特征层的box的分数放在一起
        # 放在一行里面便于操作
        boxes = tf.concat(boxes, axis=0)  # 将每个特征层的box放在一起
        box_scores = tf.concat(box_scores, axis=0)  # 将每个特征层的box的分数放在一起

        mask = box_scores >= self.obj_threshold  # 取出分数大于obj_threshold的框
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)  # 构建一个最大box数量的张量
        boxes_ = []  # 物体框的位置
        scores_ = []  # 物体类别的概率
        classes_ = []  # 物体类别

        # ---------------------------------------#
        #   1、取出每一类得分大于self.obj_threshold的框和得分
        #   2、对得分进行非极大抑制
        # ---------------------------------------#
        # 对每一个类进行判断
        for c in range(len(self.class_names)):  # 对每一个类进行判断，len(self.class_names)表示类别的数量
            # 取出所有类为c的box
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有类为c的分数
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            # 非极大抑制
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,iou_threshold=self.nms_threshold)  # 非极大抑制

            # 获取非极大抑制的结果
            class_boxes = tf.gather(class_boxes, nms_index)  # 获取非极大抑制的结果，tf.gather表示从class_boxes中取出nms_index对应的数据
            class_box_scores = tf.gather(class_box_scores, nms_index)  # 获取非极大抑制的结果，tf.gather表示从class_box_scores中取出nms_index对应的数据
            classes = tf.ones_like(class_box_scores,'int32') * c  # 获取非极大抑制的结果，tf.ones_like表示生成一个和class_box_scores相同大小的全1矩阵，再乘以c

            boxes_.append(class_boxes)  # 将每个类的box放在一起，boxes_是一个列表，每个元素是一个类的box
            scores_.append(class_box_scores)  # 将每个类的box的分数放在一起，scores_是一个列表，每个元素是一个类的box的分数
            classes_.append(classes)  # 将每个类的box的类别放在一起，classes_是一个列表，每个元素是一个类的box的类别
        boxes_ = tf.concat(boxes_, axis=0)  # 将每个类的box放在一起，boxes_是一个列表，每个元素是一个类的box
        scores_ = tf.concat(scores_, axis=0)  # 将每个类的box的分数放在一起，scores_是一个列表，每个元素是一个类的box的分数
        classes_ = tf.concat(classes_, axis=0)  # 将每个类的box的类别放在一起，classes_是一个列表，每个元素是一个类的box的类别
        return boxes_, scores_, classes_  # 返回物体框的位置、物体类别的概率、物体类别

    """
    构建预测模型。predict用于预测，分三步：1、建立yolo对象；2、获得预测结果；3、对预测结果进行处理
    输入：inputs：处理之后的输入图片；image_shape：图像原始大小
    输出：boxes：物体框坐标；scores：物体概率值；classes：物体类别
    """

    def predict(self, inputs, image_shape):  # 预测，分三步：1、建立yolo对象；2、获得预测结果；3、对预测结果进行处理
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path,
                     pre_train=False)  # 建立yolo对象，pre_train表示是否加载预训练权重，这里不加载，因为预测阶段不需要训练
        # yolo_inference用于获得网络的预测结果
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training=False)  # 获得网络的预测结果
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)  # 对预测结果进行处理
        return boxes, scores, classes  # 返回物体框的位置、物体类别的概率、物体类别