import cv2
import keras
import numpy as np
import colorsys
import pickle
import os
import network.frcnn as frcnn
from network.frcnn_training import get_new_image_size
from keras import backend as K
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image, ImageFont, ImageDraw
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from utils.config import Config
import copy
import math


class FASTERRCNN(object):
    def __init__(self):
        self.__dict__.update(self.defaults)
        self.class_names = self._get_class
        self.sess = K.get_session()
        self.config = Config()
        self.generate()
        self.bbox_util = BBoxUtility()

    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            filter_sizes = [7, 3, 1, 1]
            padding = [3, 1, 0, 0]
            stride = 2
            for i in range(4):
                input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
            return input_length

        return get_output_length(width), get_output_length(height)

        # ---------------------------------------------------#
        #   获得所有的分类
        # ---------------------------------------------------#

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算总的种类
        self.num_classes = len(self.class_names) + 1

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.model_rpn, self.model_classifier = frcnn.get_predict_model(self.config, self.num_classes)
        self.model_rpn.load_weights(self.model_path, by_name=True)
        self.model_classifier.load_weights(self.model_path, by_name=True, skip_mismatch=True)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        old_width = image_shape[1]
        old_hight = image_shape[0]
        old_image = copy.deepcopy(image)
        resize_width, resize_height = get_new_image_size(old_width, old_hight)

        image = image.resize([resize_width, resize_height])
        photo = np.array(image, dtype=np.float64)
        # 图片预处理，归一化
        photo = preprocess_input(np.expand_dims(photo, 0))
        preds = self.model_rpn.predict(photo)
        # 将预测结果进行解码
        anchors = get_anchors(self.get_img_output_length(resize_width, resize_height), resize_width, resize_height)

        rpn_reslut = self.bbox_util.detection_out(preds, anchors, 1, confidence_threshold=0)
        R = rpn_reslut[0][:, 2:]

        R[:, 0] = np.array(np.round(R[:, 0] * resize_width / self.config.rpn_stride), dtype=np.int32)
        R[:, 1] = np.array(np.round(R[:, 1] * resize_height / self.config.rpn_stride), dtype=np.int32)
        R[:, 2] = np.array(np.round(R[:, 2] * resize_width / self.config.rpn_stride), dtype=np.int32)
        R[:, 3] = np.array(np.round(R[:, 3] * resize_height / self.config.rpn_stride), dtype=np.int32)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        base_layer = preds[2]

        delete_line = []
        for i, r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)
        R = np.delete(R, delete_line, axis=0)

        bboxes = []
        probs = []
        labels = []

        for j in range(R.shape[0] // self.config.num_rois + 1):
            RIOs = np.expand_dims(R[self.config.num_rois * j: self.config.num_rois * (j + 1), :], axis=0)

            if RIOs.shape[1] == 0:
                break

            if j == R.shape[0] // self.config.num_rois:
                curr_shape = RIOs.shape
                target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
                RIOspadded = np.zeros(target_shape).astype(RIOs.dtype)
                RIOspadded[:, :curr_shape[1], :] = RIOs
                RIOspadded[0, curr_shape[1]:, :] = RIOs[0, 0, :]
                RIOs = RIOspadded

            [P_cls, P_regr] = self.model_classifier.predict([base_layer, RIOs])

            for i in range(P_cls.shape[1]):
                if np.max(P_cls[0, i, :]) < self.confidence or np.argmax(P_cls[0, i, :]) == (P_cls.shape[2] - 1):
                    continue

                label = np.argmax(P_cls[0, i, :])
                (x, y, w, h) = RIOs[0, i, :]
                cls_num = np.argmax(P_cls[0, i, :])

                (tx, ty, tw, th) = P_regr[0, i, 4 * cls_num:4 * (cls_num + 1)]
                tx /= self.config.classifier_regr_std[0]
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]

                cx = x + w / 2.
                cy = y + h / 2
                cx1 = tx * w + cx
                cy1 = ty * h + cy

                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1 / 2
                y1 = cy1 - h1 / 2

                x2 = cx1 + w1 / 2
                y2 = cy1 + h1 / 2

                x1 = int(round(x1))
                x2 = int(round(x2))
                y1 = int(round(y1))
                y2 = int(round(y2))

                bboxes.append([x1, y1, x2, y2])
                probs.append(np.max(P_cls[0, i, :]))
                labels.append(label)

        if len(bboxes) == 0:
            return old_image

        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes, dtype=np.float32)
        boxes[:, 0] = bboxes[:, 0] * self.config.rpn_stride / resize_width
        boxes[:, 1] = bboxes[:, 1] * self.config.rpn_stride / resize_height
        boxes[:, 2] = bboxes[:, 2] * self.config.rpn_stride / resize_width
        boxes[:, 3] = bboxes[:, 3] * self.config.rpn_stride / resize_height

        results = np.array(
            self.bbox_util.nms_for_out(np.array(labels), np.array(probs), np.array(boxes), self.num_classes - 1,
                                       0.4))

        top_one = results[:, 0]
        top_two = results[:, 1]
        top_three = results[:, 2:]
        boxes[:, 0] = boxes[:, 0] * old_width
        boxes[:, 1] = boxes[:, 1] * old_hight
        boxes[:, 2] = boxes[:, 2] * old_width
        boxes[:, 3] = boxes[:, 3] * old_hight

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // resize_width
        image = old_image

        for i, c in enumerate(top_one):
            predicted_class = self.class_names[int(c)]
            score = top_two[i]

            left, top, right, bottom = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right+5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left+0.5).astype('int32'))

            bottom =min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right =min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def close_session(self):
        self.sess.close()




