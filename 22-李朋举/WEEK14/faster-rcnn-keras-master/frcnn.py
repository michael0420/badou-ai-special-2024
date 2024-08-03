import cv2
import keras
import numpy as np
import colorsys
import pickle
import os
import nets.frcnn as frcnn
from nets.frcnn_training import get_new_img_size
from keras import backend as K
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image, ImageFont, ImageDraw
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from utils.config import Config
import copy
import math


class FRCNN(object):

    # ---------------------------------------------------#
    #   0. 定义模型文件、类别文件路径和置信度阈值
    # ---------------------------------------------------#
    _defaults = {
        "model_path": 'model_data/voc_weights.h5',     # 模型文件的路径
        "classes_path": 'model_data/voc_classes.txt',  # 类别文件的路径
        "confidence": 0.7,  # 置信度阈值，默认值为 0.7。置信度阈值用于判断模型的预测结果是否可靠
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化faster RCNN
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)    # 0. 定义模型文件、类别文件路径和置信度阈值
        self.class_names = self._get_class()    # 1. 获得所有的分类
        self.sess = K.get_session()
        self.config = Config()  # 2. 定义RPN模型的相关参数
        self.generate()  # 3. 获得所有的分类
        self.bbox_util = BBoxUtility()  # 4. 初始化BBoxregr的相关参数

    # ---------------------------------------------------#
    #   1. 获得所有的分类
    # ---------------------------------------------------#
    # 从指定的文件中读取类别名称，并返回一个经过处理的列表
    def _get_class(self):
        # 使用 os.path.expanduser 函数将 self.classes_path 中的路径扩展为用户的主目录。这样可以处理包含 ~ 符号的路径，例如 ~/Documents/model_classes.txt。
        classes_path = os.path.expanduser(self.classes_path)  # 'model_data/voc_classes.txt'
        # 使用 with 语句打开文件，并将文件对象赋值给变量 f。这样可以确保文件在使用后被正确关闭，避免了资源泄漏的问题。
        with open(classes_path) as f:
            # 读取文件中的所有行，并将它们存储在列表 class_names 中。
            # {list:20} ['aeroplane\n', 'bicycle\n', 'bird\n', 'boat\n', 'bottle\n', 'bus\n', 'car\n', 'cat\n', 'chair\n', 'cow\n', 'diningtable\n', 'dog\n', 'horse\n', 'motorbike\n', 'person\n', 'pottedplant\n', 'sheep\n', 'sofa\n', 'train\n', 'tvmonitor']
            class_names = f.readlines()
        # 使用列表推导式对 class_names 列表中的每个元素进行处理。c.strip() 用于去除每行的前后空格和换行符，从而得到干净的类别名称
        #    {list:20}  ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   3. 获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)  # 'model_data/voc_weights.h5'
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算总的种类
        self.num_classes = len(self.class_names) + 1  # 21

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。否则先构建模型再载入
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

    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            # input_length += 6
            filter_sizes = [7, 3, 1, 1]
            padding = [3, 1, 0, 0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
            return input_length

        return get_output_length(width), get_output_length(height)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])  # 获取输入图像的形状  # [1330 1330]
        old_width = image_shape[1]  # 原始图像的宽度 1330
        old_height = image_shape[0]  # 原始图像的高度 1330
        old_image = copy.deepcopy(image) # 复制原始图像
        width, height = get_new_img_size(old_width, old_height) # 获取调整后的图像宽度和高度 600,600

        image = image.resize([width, height])  # 将图像调整为新的尺寸  <PIL.Image.Image image mode=RGB size=600x600 at 0x1FCB5C4D860>
        photo = np.array(image, dtype=np.float64)  # 将调整后的图像转换为 numpy 数组 (600,600,3)

        # 图片预处理，归一化
        photo = preprocess_input(np.expand_dims(photo, 0))  # (1,600,600,3)
        # 使用模型的区域提议网络（RPN）对图像进行预测
        preds = self.model_rpn.predict(photo)
        # 将预测结果进行解码
        anchors = get_anchors(self.get_img_output_length(width, height), width, height)  # {12996,4}
        # 进行边界框检测
        rpn_results = self.bbox_util.detection_out(preds, anchors, 1, confidence_threshold=0)
        # print(rpn_results)
        # R = rpn_results[0][:,2:]
        # 获取检测结果中的边界框坐标
        R = rpn_results[0][:, 2:] # (300,4)  [[0.07517484 0.41013243 0.16374133 0.69341194], [0.68319498 0.38469281 0.85833523 0.61035631], ...

        # 对边界框坐标进行调整，使其与原始图像的尺寸匹配
        R[:, 0] = np.array(np.round(R[:, 0] * width / self.config.rpn_stride), dtype=np.int32)
        R[:, 1] = np.array(np.round(R[:, 1] * height / self.config.rpn_stride), dtype=np.int32)
        R[:, 2] = np.array(np.round(R[:, 2] * width / self.config.rpn_stride), dtype=np.int32)
        R[:, 3] = np.array(np.round(R[:, 3] * height / self.config.rpn_stride), dtype=np.int32)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        base_layer = preds[2]

        # 创建一个列表，用于存储要删除的边界框索引
        delete_line = []
        # 检查边界框的宽度和高度是否小于 1，如果是，则将其索引添加到 delete_line 列表中
        for i, r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)
        # 根据 delete_line 列表删除无效的边界框
        R = np.delete(R, delete_line, axis=0)

        # 创建空列表，用于存储检测到的边界框、置信度和类别标签
        bboxes = []
        probs = []
        labels = []
        # 通过循环处理每个感兴趣区域（ROI）
        for jk in range(R.shape[0] // self.config.num_rois + 1):
            # 获取当前 ROI
            ROIs = np.expand_dims(R[self.config.num_rois * jk:self.config.num_rois * (jk + 1), :], axis=0)

            # 如果 ROI 为空，则中断循环
            if ROIs.shape[1] == 0:
                break

            # 如果当前 ROI 是最后一个，则进行填充，以确保与预期的 ROI 数量一致
            if jk == R.shape[0] // self.config.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier.predict([base_layer, ROIs])

            # 对于每个类别，检查其置信度是否大于阈值，如果是，则将其信息添加到相应的列表中
            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < self.confidence or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                label = np.argmax(P_cls[0, ii, :])

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])

                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= self.config.classifier_regr_std[0]
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]

                cx = x + w / 2.
                cy = y + h / 2.
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1 / 2.
                y1 = cy1 - h1 / 2.

                x2 = cx1 + w1 / 2
                y2 = cy1 + h1 / 2

                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                bboxes.append([x1, y1, x2, y2])
                probs.append(np.max(P_cls[0, ii, :]))
                labels.append(label)

        # 如果没有检测到边界框，则直接返回原始图像
        if len(bboxes) == 0:
            return old_image

        # 对检测结果进行非极大值抑制（NMS）操作，以去除重叠的边界框
        # 筛选出其中得分高于confidence的框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes, dtype=np.float32)
        boxes[:, 0] = boxes[:, 0] * self.config.rpn_stride / width
        boxes[:, 1] = boxes[:, 1] * self.config.rpn_stride / height
        boxes[:, 2] = boxes[:, 2] * self.config.rpn_stride / width
        boxes[:, 3] = boxes[:, 3] * self.config.rpn_stride / height
        # 获取 NMS 操作后的顶级标签索引、置信度和边界框
        results = np.array(
            self.bbox_util.nms_for_out(np.array(labels), np.array(probs), np.array(boxes), self.num_classes - 1, 0.4))

        top_label_indices = results[:, 0]
        top_conf = results[:, 1]
        boxes = results[:, 2:]
        boxes[:, 0] = boxes[:, 0] * old_width
        boxes[:, 1] = boxes[:, 1] * old_height
        boxes[:, 2] = boxes[:, 2] * old_width
        boxes[:, 3] = boxes[:, 3] * old_height

        # 据顶级标签索引获取对应的类别名称和颜色
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // width
        image = old_image
        # 遍历顶级标签索引，绘制边界框和类别标签
        # 调整边界框的坐标，使其与原始图像的尺寸匹配
        # 绘制矩形框，并根据类别设置颜色
        # 在边界框上方绘制类别标签和置信度
        # 返回绘制了边界框和类别标签的图像
        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]


            left, top, right, bottom = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

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
