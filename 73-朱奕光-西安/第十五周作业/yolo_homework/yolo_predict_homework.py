import os
import numpy as np
import colorsys
import random
from model.yolo3_model import yolo
import config_homework as config
import tensorflow as tf

class yolo_predictor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.classes_path = classes_file
        self.anchors_path = anchors_file
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        # 定义每种类别的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]

        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)


    def _get_class(self):
        """
        :return: 类别名称 （列表类型）
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        利用K-means设置先验框，加快收敛速度
        :return: 先验框坐标
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(n) for n in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        确定Bounding box相关信息
        :param feats:  yolo模型的输出
        :param anchors:  读取的先验框位置
        :param num_classes:  类别数量
        :param input_shape:  输入大小
        :return: box_xy： box的坐标
                 box_wh： box的宽高
                 box_confidence： box的置信度
                 box_class_probs： box对应的每个类别的概率
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)

        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        计算物体框的预测坐标在原图中的位置坐标
        :param box_xy:  物体框左上角坐标
        :param box_wh:  物体框宽高
        :param input_shape:  输入的大小
        :param image_shape:  原图的大小
        :return: 对应在原始图像上的box坐标，一个张量，尺寸为[num_boxes，4]，4为左上角的xy坐标和右下角的xy坐标
        """
        # 翻转坐标和长宽的顺序
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        input_shape = tf.cast(input_shape, dtype=tf.float32)
        image_shape = tf.cast(image_shape, dtype=tf.float32)

        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    def boxes_and_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        """
        将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        :param feats:  yolo输出的结果
        :param anchors:  预选框的位置
        :param num_classes:  类别数量
        :param input_shape:  输入尺寸
        :param image_shape:  原图尺寸
        :return:  box：物体框的位置坐标
                  boxes：物体框的分数，为置信度和每个类别概率的乘积
        """
        # 获得特征
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, num_classes, input_shape)
        #
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, num_classes])
        return boxes, box_scores

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        """
        对yolo模型结果进行与之筛选和NMS， 获取最后的物体框和物体框类别
        :param yolo_outputs:  yolo模型结果
        :param image_shape:  原图尺寸
        :param max_boxes:  最大box数量
        :return:  boxes_： 物体框坐标
                  scores_： 物体类别的概率
                  classes_ ： 物体类别
        """
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32

        """
        对三个特征层的输出解码，获取每个物体框的位置和分数
        """
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]], len(self.class_names),
                                                        input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        """
        先用阈值obj_threshold进行一轮筛选
        后用NMS进行非极大值抑制
        """
        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []

        # 对每一类进行判断
        for c in range(len(self.class_names)):
            # 取出所有类别为c的box
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            #取出所有类别为c的box的得分
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            # NMS
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=self.nms_threshold)
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)
        return boxes_, scores_, classes_

    def predict(self, inputs, image_shape):
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train=False)
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training=False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)
        return boxes, scores, classes
