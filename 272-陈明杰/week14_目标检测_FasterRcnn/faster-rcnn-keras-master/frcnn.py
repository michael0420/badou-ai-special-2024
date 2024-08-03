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
    _defaults = {
        "model_path": 'model_data/voc_weights.h5',
        "classes_path": 'model_data/voc_classes.txt',
        "confidence": 0.7,
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
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.config = Config()
        self.generate()
        self.bbox_util = BBoxUtility()

    # # ---------------------------------------------------#
    # #   获得所有的分类
    # # ---------------------------------------------------#
    # def _get_class(self):
    #     classes_path = os.path.expanduser(self.classes_path)
    #     with open(classes_path) as f:
    #         class_names = f.readlines()
    #     class_names = [c.strip() for c in class_names]
    #     return class_names
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # def generate(self):
    #     model_path = os.path.expanduser(self.model_path)
    #     assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
    #
    #     # 计算总的种类
    #     self.num_classes = len(self.class_names) + 1
    #
    #     # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
    #     # 否则先构建模型再载入
    #     self.model_rpn, self.model_classifier = frcnn.get_predict_model(self.config, self.num_classes)
    #     self.model_rpn.load_weights(self.model_path, by_name=True)
    #     self.model_classifier.load_weights(self.model_path, by_name=True, skip_mismatch=True)
    #
    #     print('{} model, anchors, and classes loaded.'.format(model_path))
    #
    #     # 画框设置不同的颜色
    #     hsv_tuples = [(x / len(self.class_names), 1., 1.)
    #                   for x in range(len(self.class_names))]
    #     self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    #     self.colors = list(
    #         map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
    #             self.colors))
    def generate(self):
        # model_path=os.path.expanduser(self.model_path)
        # assert model_path.endswith('.h5'),'Keras model or weights must be a .h5 file.'
        self.num_classes = len(self.class_names) + 1
        self.model_rpn, self.model_classifier = frcnn.get_predict_model(self.config, self.num_classes)
        self.model_rpn.load_weights(self.model_path, by_name=True, skip_mismatch=True)
        self.model_classifier.load_weights(self.model_path, by_name=True, skip_mismatch=True)
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                               self.colors))

    # def get_img_output_length(self, width, height):
    #     def get_output_length(input_length):
    #         # input_length += 6
    #         filter_sizes = [7, 3, 1, 1]
    #         padding = [3, 1, 0, 0]
    #         stride = 2
    #         for i in range(4):
    #             # input_length = (input_length - filter_size + stride) // stride
    #             input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
    #         return input_length
    #     return get_output_length(width), get_output_length(height)
    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            filter_sizes = [7, 3, 1, 1]
            padding = [3, 1, 0, 0]
            stride = 2
            for i in range(4):
                input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // 2 + 1
            return input_length

        return get_output_length(width), get_output_length(height)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # image_shape = np.array(np.shape(image)[0:2])
        image_shape = np.array(np.shape(image)[0:2])
        # old_width = image_shape[1]
        old_width = image_shape[1]
        # old_height = image_shape[0]
        old_height = image_shape[0]
        # old_image = copy.deepcopy(image)
        old_image = copy.deepcopy(image)
        # width, height = get_new_img_size(old_width, old_height)
        width, height = get_new_img_size(old_width, old_height)

        # image = image.resize([width, height])
        image = image.resize([width, height])
        # photo = np.array(image, dtype=np.float64)
        photo = np.array(image, dtype=np.float64)

        # 图片预处理，归一化
        # photo = preprocess_input(np.expand_dims(photo, 0))
        photo = preprocess_input(np.expand_dims(photo, 0))
        # preds = self.model_rpn.predict(photo)
        preds = self.model_rpn.predict(photo)
        # 将预测结果进行解码
        # anchors = get_anchors(self.get_img_output_length(width, height), width, height)
        anchors = get_anchors(self.get_img_output_length(width, height), width, height)
        # 每一个先验框对应于每一个预测结果[dx,dy,dw,dh]，对所有的先验框调整并按照一个规则
        # 初步筛选后的结果，bbox_util.detection_out函数之后得到的是经过调整和筛选后的框了，
        # 已经算是比较准确的框了
        # rpn_results = self.bbox_util.detection_out(preds, anchors, 1, confidence_threshold=0)

        # RPN阶段的非极大值抑制：
        # 在RPN（Region Proposal Network）阶段，NMS的主要目的是从大量的候选区域（anchors经过分类和回归
        # 后得到的proposals）中筛选出最有可能包含物体的框。这些候选区域是基于特征图上的锚点（anchors）生成
        # 的，并经过边界框回归和分类得分的预测。
        # 在这个过程中，NMS会根据候选区域的分类得分（通常是物体存在的概率）和它们之间的重叠度（IoU）来去除冗
        # 余的候选框。具体来说，它会选择得分最高的候选框作为参考，并去除那些与参考框重叠度较高但得分较低的候选
        # 框。这样，每个可能包含物体的位置只保留一个得分最高的候选框，从而减少候选框的数量，提高后续处理的效率。
        # 分类阶段的非极大值抑制：
        # 在分类阶段（通常是指Fast R-CNN部分或类似的检测头），NMS的作用是去除对同一个物体的冗余检测框。在这个
        # 阶段，候选区域已经经过了更精细的分类和位置回归，每个候选区域都被赋予了一个具体的类别预测和对应的得分。
        # 由于在图像中可能存在多个物体，并且这些物体可能相互靠近或重叠，因此分类阶段可能会生成多个对同一个物体
        # 的检测框。NMS会根据这些检测框的类别得分和它们之间的重叠度来去除冗余的框，确保每个物体只被一个最优的
        # 检测框所表示。
        # 总的来说，RPN阶段的NMS侧重于从大量候选区域中筛选出最有可能包含物体的框，而分类阶段的NMS则侧重于去除
        # 对同一个物体的冗余检测框，从而提高检测的精度和效率。
        # 这里面的候选框是筛选出框中具有物体的候选框
        rpn_results = self.bbox_util.detection_out(preds, anchors, 1)
        # R = rpn_results[0][:, 2:]
        # R表示这张图片所有的框的[x,y,w,h]
        R = rpn_results[0][:, 2:]
        # 因为我们的先验框是按照原图的尺寸生成的，所以需要把这些先验框放大到feature_map尺度下的尺寸大小
        # R[:, 0] = np.array(np.round(R[:, 0] * width / self.config.rpn_stride), dtype=np.int32)
        # R[:, 1] = np.array(np.round(R[:, 1] * height / self.config.rpn_stride), dtype=np.int32)
        # R[:, 2] = np.array(np.round(R[:, 2] * width / self.config.rpn_stride), dtype=np.int32)
        # R[:, 3] = np.array(np.round(R[:, 3] * height / self.config.rpn_stride), dtype=np.int32)
        R[:, 0] = np.array(np.round(R[:, 0] * width / self.config.rpn_stride), dtype=np.int32)
        R[:, 1] = np.array(np.round(R[:, 1] * height / self.config.rpn_stride), dtype=np.int32)
        R[:, 2] = np.array(np.round(R[:, 2] * width / self.config.rpn_stride), dtype=np.int32)
        R[:, 3] = np.array(np.round(R[:, 3] * height / self.config.rpn_stride), dtype=np.int32)

        # R[:, 2] -= R[:, 0]
        # R[:, 3] -= R[:, 1]
        # 计算出长宽
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # base_layer = preds[2]
        base_layer = preds[2]

        # delete_line = []
        delete_line = []
        # for i, r in enumerate(R):
        #     if r[2] < 1 or r[3] < 1:
        #         delete_line.append(i)
        # R = np.delete(R, delete_line, axis=0)
        for i, r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)
        R = np.delete(R, delete_line, axis=0)

        bboxes = []
        probs = []
        labels = []
        for jk in range(R.shape[0] // self.config.num_rois + 1):
            ROIs = np.expand_dims(R[self.config.num_rois * jk:self.config.num_rois * (jk + 1), :], axis=0)

            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // self.config.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            # 分类和回归，就是rpn之后的再一次的调整预测框，并对每一个预测框的内容进行分类
            # base_layer就是特征图，ROIs就是rpn得到的预测框，两者结合起来得到每一个框对应的特征图
            [P_cls, P_regr] = self.model_classifier.predict([base_layer, ROIs])

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

                # 预测框
                bboxes.append([x1, y1, x2, y2])
                # 预测框中内容的类别置信度
                probs.append(np.max(P_cls[0, ii, :]))
                # 类别标签
                labels.append(label)

        if len(bboxes) == 0:
            return old_image

        # 筛选出其中得分高于confidence的框，其实也就是去除冗余框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes, dtype=np.float32)
        boxes[:, 0] = boxes[:, 0] * self.config.rpn_stride / width
        boxes[:, 1] = boxes[:, 1] * self.config.rpn_stride / height
        boxes[:, 2] = boxes[:, 2] * self.config.rpn_stride / width
        boxes[:, 3] = boxes[:, 3] * self.config.rpn_stride / height
        # 这里的非极大值抑制是去除冗余的框
        results = np.array(
            self.bbox_util.nms_for_out(np.array(labels), np.array(probs), np.array(boxes), self.num_classes - 1, 0.4))

        top_label_indices = results[:, 0]
        top_conf = results[:, 1]
        boxes = results[:, 2:]
        boxes[:, 0] = boxes[:, 0] * old_width
        boxes[:, 1] = boxes[:, 1] * old_height
        boxes[:, 2] = boxes[:, 2] * old_width
        boxes[:, 3] = boxes[:, 3] * old_height

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // width
        image = old_image
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

    # # 检测图片的函数定义
    # def detect_image(self, image):
    #     # 获取图像的高度和宽度（假设image是PIL图像或类似对象，这里直接取shape可能不适用，但这里用numpy的shape模拟）
    #     image_shape = np.array(image.size)  # 更常见的PIL图像尺寸获取方式
    #     old_width, old_height = image_shape  # 解构赋值
    #
    #     # 复制原始图像以便后续处理（保持原始图像不变）
    #     old_image = image.copy()  # 对于PIL图像，使用copy()方法复制
    #
    #     # 获取调整大小后的图像尺寸（这里假设get_new_img_size函数根据某种策略返回新尺寸）
    #     width, height = get_new_img_size(old_width, old_height)
    #
    #     # 调整图像大小（这里假设image对象有resize方法，如PIL图像）
    #     image = image.resize((width, height))
    #
    #     # 将PIL图像转换为NumPy数组，并进行数据类型转换（通常这一步在预处理函数中完成，但为了清晰展示）
    #     # 这里跳过直接转换，因为preprocess_input可能期望特定的输入格式
    #
    #     # 对图像进行预处理（如归一化、数据增强等），并增加一个批次维度
    #     # 假设preprocess_input函数接受NumPy数组并返回预处理后的批次数据
    #     # 注意：这里我们实际上没有从PIL图像到NumPy数组的转换，但在实际应用中需要
    #     photo = preprocess_input(np.expand_dims(np.array(image), 0))  # 这里假设了额外的步骤来转换图像
    #
    #     # 使用RPN模型进行预测（假设model_rpn是一个已经训练好的RPN模型）
    #     preds = self.model_rpn.predict(photo)
    #
    #     # 获取锚框（先验框），这些是根据图像尺寸和网络配置生成的固定框
    #     anchors = get_anchors(self.get_img_output_length(width, height), width, height)
    #
    #     # 对RPN的预测结果进行解码，获取候选区域（建议框）
    #     rpn_results = self.bbox_util.detection_out(preds, anchors, 1, confidence_threshold=0)
    #     # 注意：这里的confidence_threshold=0可能不是实际使用中的好选择，因为它会保留所有候选区域
    #
    #     # 提取候选区域的坐标（通常包括中心点坐标和宽高）
    #     R = rpn_results[0][:, 2:]  # 假设检测结果的第一部分是坐标，且索引从2开始
    #
    #     # 将候选区域的坐标从模型输出空间转换回原始图像空间
    #     # 这里假设RPN的输出坐标是相对于特征图的，并且需要缩放到原始图像尺寸
    #     R[:, 0] = np.round(R[:, 0] * width / self.config.rpn_stride).astype(np.int32)
    #     R[:, 1] = np.round(R[:, 1] * height / self.config.rpn_stride).astype(np.int32)
    #     R[:, 2] = np.round(R[:, 2] * width / self.config.rpn_stride).astype(np.int32)
    #     R[:, 3] = np.round(R[:, 3] * height / self.config.rpn_stride).astype(np.int32)
    #
    #     # 如果坐标表示的是中心点+宽高，则需要转换为左上角+右下角
    #     R[:, 2] = R[:, 0] + R[:, 2]  # 转换为x2
    #     R[:, 3] = R[:, 1] + R[:, 3]  # 转换为y2
    #
    #     # ...（此处可能还有其他处理，如过滤掉太小的框）
    #
    #     # 假设preds[2]是RPN网络的一个特征图输出，用于后续的分类和回归
    #     base_layer = preds[2]  # 这通常不是RPN的直接输出，但这里为了示例假设它存在
    #
    #     # 初始化列表，用于存储最终的检测结果
    #     bboxes = []
    #     probs = []
    #     labels = []
    #
    #     # 对每个候选区域进行分类和回归预测（这里简化了循环和批处理逻辑）
    #     for jk in range(0, R.shape[0], self.config.num_rois):  # 假设每次处理num_rois个候选区域
    #         end_idx = min(jk + self.config.num_rois, R.shape[0])
    #         ROIs = R[jk:end_idx]
    #
    #         # 这里通常需要对ROIs进行编码（如RoI Pooling/Align），然后输入到分类和回归网络中
    #         # 但为了简化，我们假设有一个直接的预测步骤
    #         # [P_cls, P_regr] = self.model_classifier.predict(...)  # 这里省略了详细的预测步骤
    #
    #         # 假设我们已经有了分类和回归的预测结果P_cls和P_regr（这里用模拟数据代替）
    #         # ...（模拟预测结果的代码省略）
    #
    #         # 对每个预测结果进行解码，并添加到结果列表中
    #         for ii in range(P_cls.shape[1]):  # 假设P_cls的第二维是候选区域的数量
    #             if np.max(P_cls[0, ii, :]) < self.confidence or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
    #                 continue  # 忽略得分低于阈值或背景类的候选区域
    #
    #             label = np.argmax(P_cls[0, ii, :])
    #             (x, y, w, h) = ROIs[ii]  # 注意：这里ROIs已经转换为左上角+右下角坐标
    #
    #             # 使用回归参数调整目标框
    #             # ...（这里省略了回归参数的应用，通常包括中心偏移和宽高缩放）
    #             # 假设我们已经有了调整后的目标框坐标(x1, y1, x2, y2)
    #
    #             bboxes.append([x1, y1, x2, y2])
    #             probs.append(np.max(P_cls[0, ii, :]))
    #             labels.append(label)
    #
    #             # 如果没有检测到任何目标，则返回原始图像
    #     if not bboxes:
    #         return old_image
    #
    #         # 对检测到的目标框应用NMS（非极大值抑制），去除重叠的目标框
    #     # 这里省略了NMS的具体实现，但通常它会接受类别、得分和坐标作为输入，并返回筛选后的结果
    #     # results = self.bbox_util.nms_for_out(np.array(labels), np.array(probs), np.array(bboxes), ...)
    #
    #     # 假设results已经包含了筛选后的目标框信息，并且我们已经将其转换为了适当的格式
    #     # 这里我们直接跳过NMS和后续坐标转换的步骤，因为它们的具体实现依赖于NMS函数和配置
    #
    #     # 加载字体，准备在图像上绘制文本
    #     font = ImageFont.truetype(font='model_data/simhei.ttf', size=...)  # 字体大小和路径需要根据实际情况设置
    #
    #     # 设置线条粗细（这里使用了一个简单的计算方式，但实际应用中可能需要调整）
    #     thickness = (old_height + old_width) // 300  # 假设一个基于图像尺寸的简单计算
    #
    #     # 绘制检测结果到原始图像上
    #     draw = ImageDraw.Draw(old_image)  # 在原始图像上绘制，以保持图像质量
    #     for i, c in enumerate(labels):  # 假设labels已经是筛选后的类别索引列表
    #         predicted_class = self.class_names[c]  # 将类别索引转换为类别名称
    #         score = probs[i]  # 获取对应的得分
    #
    #         # 获取目标框坐标（这里假设bboxes已经是筛选且转换后的坐标列表）
    #         x1, y1, x2, y2 = bboxes[i]
    #
    #         # 调整目标框坐标以适应绘制（如添加一些边距）
    #         # ...（这里省略了坐标调整的代码）
    #
    #         # 绘制目标框
    #         for _ in range(thickness):
    #             draw.rectangle([x1 + _, y1 + _, x2 - _, y2 - _], outline=self.colors[c])
    #
    #             # 绘制文本标签
    #         label_text = f'{predicted_class} {score:.2f}'
    #         text_width, text_height = draw.textsize(label_text, font)
    #
    #         # 计算文本标签的位置（这里简化了位置计算，实际应用中可能需要更复杂的逻辑）
    #         text_x = x1
    #         if x1 + text_width > old_width:
    #             text_x = x1 - text_width - 10
    #             text_y = y1 - text_height - 10  # 假设文本位于目标框上方
    #
    #             # 确保文本不会超出图像上方
    #             text_y = max(0, text_y)
    #
    #             # 绘制目标框
    #             for _ in range(thickness):
    #                 draw.rectangle([x1 + _, y1 + _, x2 - _, y2 - _], outline=self.colors[class_id])
    #
    #                 # 可选：绘制文本标签的背景框
    #             # draw.rectangle([text_x-2, text_y-2, text_x+text_width+2, text_y+text_height], fill=(255, 255, 255, 128))
    #
    #             # 绘制文本标签
    #             draw.text((text_x, text_y), label_text, fill=(0, 0, 0), font=font)
    #
    #     # 由于我们在原始图像上进行了绘制，所以直接返回带有检测结果的原始图像
    #     return old_image

    def close_session(self):
        self.sess.close()
