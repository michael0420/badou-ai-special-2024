import colorsys
import random
import config
import numpy as np
from model import yolo3模型实现
import tensorflow as tf


class yolo_predictor_Myclass:
    def __init__(self, obj_threshold, nms_threshold, class_file, anchors_file):
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.class_file = class_file
        self.anchors_file = anchors_file
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        #每个类别分配一个随机颜色
        hsv_tuples = [(x / len(self.class_names), 1, 1) for x in range(len(self.class_names))]
        self.color = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.color = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.color))
        random.seed(10101)
        random.shuffle(self.color)
        random.seed(None)

    def _get_class(self):
        with open(self.class_file, 'r') as f:
            class_names = f.readlines()
            class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        with open(self.anchors_file, 'r') as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def _get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchors = len(anchors)
        #print(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors,dtype=tf.float32),[1,1,1,num_anchors,2])
        #print(anchors_tensor)
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats,[-1,grid_size[0],grid_size[1],num_anchors,num_classes+5])

        a = tf.reshape(tf.range(grid_size[0]),[-1,1,1,1])
        #第二个参数是一个列表，指定了每个维度上重复的次数
        grid_y = tf.tile(a,[1,grid_size[1],1,1])
        b = tf.reshape(tf.range(grid_size[1]),[1,-1,1,1])
        grid_x = tf.tile(b, [grid_size[0], 1, 1, 1])
        grid = tf.concat((grid_x, grid_y), axis=-1)
        grid = tf.cast(grid,tf.float32)

        #predictions[...,:2]语法表示留下predictions的最后一个维度中的前两个数据
        box_xy = (tf.sigmoid(predictions[...,:2]) +grid) / tf.cast(grid_size[::-1],tf.float32)
        #表示将每个边界框宽度和高度的偏移乘以相应的锚框宽度和高度，这样就得到了每个边界框的实际宽度和高度。
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs


    def correct_boxes(self,box_xy,box_wh,input_shape,image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        #都转为浮点，后续要进行计算
        input_shape = tf.cast(input_shape, tf.float32)
        image_shape = tf.cast(image_shape, tf.float32)
        new_shape = tf.round(image_shape*tf.reduce_min(input_shape/image_shape))
        offset = (input_shape - new_shape) / 2. /input_shape
        scale = input_shape /new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ],axis=-1)
        boxes*=tf.concat([image_shape,image_shape],axis=-1)
        return boxes










    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        #将预测的框坐标转换为原图的坐标，然后计算每个box分数
        #print(anchors)
        box_xy,box_wh,box_confidence,box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)

        #寻找在原图中的位置
        boxes = self.correct_boxes(box_xy, box_wh,input_shape,image_shape)
        boxes = tf.reshape(boxes,[-1,4])

        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores,[-1,classes_num])
        return boxes, box_scores



    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        #这里的yolo_outputs输出是[conv59, conv67, conv75]，其中con59是13*13,67是26*26，75s是52*52
        #指定每个结果使用的框的索引
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []  #解码后的边界框位置
        boxes_scores = []  #解码后的边界框分数
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32

        for i in range(len(yolo_outputs)):
            _boxes,_box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchors_mask[i]], len(self.class_names), input_shape,
                                  image_shape)
            boxes.append(_boxes)
            boxes_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis=0)
        boxes_scores = tf.concat(boxes_scores, axis=0)

        mask = boxes_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes,dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ =[]
        for c in range(len(self.class_names)):
            # 取出所有类为c的box
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有类为c的分数
            class_box_scores = tf.boolean_mask(boxes_scores[:, c], mask[:, c])
            # 非极大抑制
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                     iou_threshold=self.nms_threshold)

            # 获取非极大抑制的结果
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis = 0)
        scores_ = tf.concat(scores_, axis = 0)
        classes_ = tf.concat(classes_, axis = 0)
        return boxes_, scores_, classes_

    def predict(self, inputs, image_shape):
        model = yolo3模型实现.yolo3(config.norm_epsilon, config.norm_decay, self.anchors_file, self.class_file,
                                    pre_train=True)
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training=False)
        boxes, scores, classes = self.eval(output, image_shape)
        return boxes, scores, classes
