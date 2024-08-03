import cv2
import keras
import numpy as np
import colorsys
import pickle
import os
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

import nets.frcnnbase as frcnn

'''
# 生成新图像大小
def get_new_img_size(width, height, img_min_side=600):
    """
    根据给定的宽度和高度以及最小边长来调整图像的大小
    :param width: 原始图像宽度
    :param height: 原始图像高度
    :param img_min_side: 图像最小边长
    :return:
    """
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)
    return resized_width,resized_height
'''

class FRCNN(object):
    # 定义默认属性值
    _defaults = {
        "model_path": 'model_data/voc_weights.h5',  # 预训练模型文件
        "classes_path": "model_data/voc_classes.txt",  # 定义物体类别文件
        "confidence": 0.7,  # 置信度阈值
    }
    # 将一个方法定义为类方法,可以在不创建类实例的情况下调用，只需通过类名即可。类方法的第一个参数通常是 cls，表示类本身。
    @classmethod
    def get_defaults(cls, n):
        # 该方法用于判断检查 属性名 n 是否在 默认属性字典中
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # 初始化网络: 接受任意数量的关键字参数
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # 更新默认属性字典
        self.class_names = self._get_class()  # 获取所有分类的名称，以列表形式
        self.sess = K.get_session()  # 使用后端模块创建一个tensorflow会话
        self.config = Config()  # 配置类实例，用于获取配置信息
        self.generate()
        self.bbox_util = BBoxUtility()

    # 获取所有的分类名称
    def _get_class(self):
        # 统一路径格式
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # 从以训练模型中生成预测
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算总的种类数量
        self.num_classes = len(self.class_names) + 1

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.model_rpn, self.model_classifier = frcnn.get_predict_model(self.config, self.num_classes)
        # 载入模型,by_name = True,表示按照模型结构中的名称来匹配权重，skip_mismatch = True, 表示如果权重名称不匹配，则跳过该权重的加载
        self.model_rpn.load_weights(self.model_path, by_name=True)
        self.model_classifier.load_weights(self.model_path, by_name=True, skip_mismatch=True)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        '''
        在 HSV 颜色模型中，每个颜色由三个参数确定：色调（hue）、饱和度（saturation）和值（value），分别简称 H、S、V。色调表示颜色的种类，如红色、蓝色等；饱和度表示颜色纯度，0 为灰度，1 为最纯的颜色；值则表示颜色的明暗程度，0 为最暗，1 为最亮。
        所以，(x / len(self.class_names), 1., 1.) 表示将色调设为从 0 到 len(self.class_names) 的范围，而饱和度和值都设为最大值 1.0，从而得到不同类别的颜色。
        '''

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # 用于计算经过卷积神经网络（CNN）处理后的图像尺寸
    def get_img_output_length(self, width, height):

        def get_output_length(input_length):
            filter_sizes = [7, 3, 1, 1]   # 4层：7*7 3*3 1*1 1*1
            padding = [3, 1, 0, 0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
            return input_length

        return get_output_length(width), get_output_length(height)  # 获得图片的长宽

    # 检测图片
    def detect_image(self,image):
    # 获取图片的输入形状的前2维度信息：高度和宽度
        image_shape = np.array(np.shape(image)[0:2])
        old_width = image_shape[1]
        old_height = image_shape[0]
        old_image = copy.deepcopy(image)  # 对图像进行深拷贝
        # 调整图像大小
        width, height = get_new_img_size(old_width, old_height)
        image = image.resize([width, height]) #新的统一的尺寸
        # 图像预处理
        photo = np.array(image, dtype=np.float64)  # 转为numpy数组，64位浮点数
        photo = preprocess_input(np.expand_dims(photo,0))  # np数组增加维度，并进行预处理
        # 使用预训练的RPN模型进行预测
        preds = self.model_rpn.predict(photo)

        # 对预测结果进行解码
        anchors = get_anchors(self.get_img_output_length(width,height),width,height)
        rpn_results = self.bbox_util.detection_out(preds, anchors, 1,confidence_threshold=0) #使用 bbox_util 处理预测输出。
        # 提取预测框的坐标信息  坐标和分数
        R = rpn_results[0][:, 2:]
        # 对边界框的坐标进行缩放和平移

        R[:, 0] = np.array(np.round(R[:, 0] * width / self.config.rpn_stride),
                           dtype=np.int32)  # 将预测框的坐标转换成原图的坐标，R[:0]表示x1
        R[:, 1] = np.array(np.round(R[:, 1] * height / self.config.rpn_stride),
                           dtype=np.int32)  # 将预测框的坐标转换成原图的坐标，R[:1]表示y1
        R[:, 2] = np.array(np.round(R[:, 2] * width / self.config.rpn_stride),
                           dtype=np.int32)  # 将预测框的坐标转换成原图的坐标，R[:2]表示x2
        R[:, 3] = np.array(np.round(R[:, 3] * height / self.config.rpn_stride),
                           dtype=np.int32)  # 将预测框的坐标转换成原图的坐标，R[:3]表示y2

        R[:, 2] -= R[:, 0]  # R[:2]表示x2-x1，即宽度
        R[:, 3] -= R[:, 1]  # 将预测框的坐标转换成原图的坐标，R[:3]表示y2-y1
        base_layer = preds[2]  # 获得预测结果中的第三个结果，即feature map

        delete_line = []  # 删除一些不合规的框
        for i, r in enumerate(R):  # 遍历所有的预测框
            if r[2] < 1 or r[3] < 1:  # 如果预测框的宽度或者高度小于1
                delete_line.append(i)  # 将这个预测框的索引添加到delete_line中
        R = np.delete(R, delete_line, axis=0)  # 删除这些不合规的预测框

        bboxes = []  # 用于存储筛选后的边界框
        probs = []   # 用于存储筛选后的概率值
        labels = []  # 用于存储筛选后的标签值
        # 遍历所有筛选的边界框
        for jk in range(R.shape[0] // self.config.num_rois + 1):
            # 将筛选后的边界框扩展为一个新的数组
            ROIs = np.expand_dims(R[self.config.num_rois*jk: self.config.num_rois*(jk+1), :],axis=0)
            # 如果筛选后的边界框数量为0，则跳出循环
            if ROIs.shape[1] == 0:
                break

            # 如果已经处理完所有筛选后的边界框，则执行以下操作
            if jk == R.shape[0] // self.config.num_rois:
                curr_shape = ROIs.shape  # 获取当前筛选后的边界框的形状。
                # 计算目标形状。
                target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)  # 创建一个全零数组，用于存储填充后的筛选后的边界框。
                ROIs_padded[:, :curr_shape[1], :] = ROIs  # 将原始筛选后的边界框复制到填充后的数组中。
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]  # 将第一个筛选后的边界框复制到填充后的数组中。
                ROIs = ROIs_padded  # 更新筛选后的边界框数组。
            # 使用分类器模型对填充后的筛选后的边界框进行预测
            [P_cls, P_regr] = self.model_classifier.predict([base_layer, ROIs])
            # 遍历所有预测类别
            for ii in range(P_cls.shape[1]):
                # 如果预测概率小于置信度阈值或预测类别为背景类别，则跳过当前循环。
                if np.max(P_cls[0, ii, :]) < self.confidence or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                label = np.argmax(P_cls[0, ii, :])  # 获取预测类别的最大值对应的索引作为标签。
                (x, y, w, h) = ROIs[0, ii, :]  # 获取筛选后的边界框的坐标和尺寸。
                cls_num = np.argmax(P_cls[0, ii, :])  # 获取预测类别的最大值对应的索引作为类别编号。
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num: 4 * (cls_num + 1)]  # 获取预测边界框的偏移量。
                # 对预测边界框的偏移量进行标准化。
                tx /= self.config.classifier_regr_std[0]  # 进行反标准化，这里的标准化是在训练的时候进行的
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]

                cx = x + w / 2.  # 计算边界框的中心点的x坐标。
                cy = y + h / 2.  # 计算边界框的中心点的y坐标。
                cx1 = tx * w + cx  # 计算预测边界框的中心点的x坐标。
                cy1 = ty * h + cy  # 计算预测边界框的中心点的y坐标。
                w1 = math.exp(tw) * w  # 计算预测边界框的宽度。
                h1 = math.exp(th) * h  # 计算预测边界框的高度。
                x1 = cx1 - w1 / 2.  # 计算预测边界框左上角的x坐标。
                y1 = cy1 - h1 / 2.  # 计算预测边界框左上角的y坐标。
                x2 = cx1 + w1 / 2  # 计算预测边界框右下角的x坐标。
                y2 = cy1 + h1 / 2  # 计算预测边界框右下角的y坐标。
                # 将预测边界框的坐标四舍五入并转换为整数。
                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))
                bboxes.append([x1, y1, x2, y2])  # 将预测边界框的坐标添加到列表中。
                probs.append(np.max(P_cls[0, ii, :]))  # 将预测概率添加到列表中。
                labels.append(label)  # 将预测类别添加到列表中。
            # 如果筛选后的边界框数量为0，则返回原始图像
        if len(bboxes) == 0:
            return old_image
            # 将标签、概率、边界框列表转换为NumPy数组。
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes, dtype=np.float32)
        # 将边界框的x、y、w、h坐标进行缩放和平移。
        boxes[:, 0] = boxes[:, 0] * self.config.rpn_stride / width
        boxes[:, 1] = boxes[:, 1] * self.config.rpn_stride / height
        boxes[:, 2] = boxes[:, 2] * self.config.rpn_stride / width
        boxes[:, 3] = boxes[:, 3] * self.config.rpn_stride / height
        # 调用了self.bbox_util.nms_for_out方法，对输入的标签、概率和边界框进行非极大值抑制（NMS），并返回结果
        results = np.array(self.bbox_util.nms_for_out(np.array(labels), np.array(probs), np.array(boxes), self.num_classes - 1, 0.4))

        top_label_indices = results[:, 0]
        top_conf = results[:, 1]
        boxes = results[:, 2:]
        boxes[:, 0] = boxes[:, 0] * old_width
        boxes[:, 1] = boxes[:, 1] * old_height
        boxes[:, 2] = boxes[:, 2] * old_width
        boxes[:, 3] = boxes[:, 3] * old_height
        # 创建一个字体对象，用于绘制文本
        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2*np.shape(image)[1]+0.5).astype('int32'))
        # 计算线条粗细
        thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // width
        image = old_image  # 将原始图像赋值给image
        # 遍历top_label_indices中的每个元素及其索引
        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]  # 根据索引获取预测的类别名称。
            score = top_conf[i]  # 获取置信度得分。
            left, top, right, bottom = boxes[i]  #从boxes中提取边界框的坐标。
            top = top - 5  # 将边界框的顶部坐标减去5。
            left = left - 5  # 将边界框的左侧坐标减去5。
            bottom = bottom + 5  # 将边界框的底部坐标加上5。
            right = right + 5  # 将边界框的右侧坐标加上5。

            top = max(0, np.floor(top + 0.5).astype('int32'))  # 确保边界框的顶部坐标不小于0。
            left = max(0, np.floor(left + 0.5).astype('int32'))  #确保边界框的左侧坐标不小于0。
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))  # 确保边界框的底部坐标不大于图像的高度。
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))  # 确保边界框的右侧坐标不大于图像的宽度。
            # 格式化预测的类别和置信度得分。
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)  # 创建一个绘图对象。
            label_size = draw.textsize(label, font)  # 计算文本的大小。
            label = label.encode('utf-8')  # 将文本编码为UTF-8格式。
            print(label)  # 打印文本。
            # 判断文本是否超出边界框的顶部。
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])  # 计算文本的起始坐标。
            # 如果超出边界框的顶部。
            else:
                # 计算文本的起始坐标。
                text_origin = np.array([left, top + 1])
            # 循环绘制线条
            for i in range(thickness):
                # 绘制矩形边框
                draw.rectangle(
                    [left+i, top+i, right-i, bottom-i],
                    outline=self.colors[int(c)])
            # 绘制文本背景
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin+label_size)],
                fill=self.colors[int(c)])
            # 绘制文本
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw  # 删除绘图对象。
        return image  # 返回处理后的图像。

    # 关闭会话
    def close_session(self):
        self.sess.close()


















































