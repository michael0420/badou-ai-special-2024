import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, PReLU, Dense, Permute, Flatten
from keras.models import Model
from utils import calculateScales, detect_face_12net, NMS, filter_face_24net, filter_face_48net
import cv2 as cv
import numpy as np


def create_Pnet(weight_path):
    input = Input(shape=[None, None, 3])
    x = Conv2D(10, kernel_size=(3, 3), strides=1, padding='valid', name='conv1')(input)
    # 对于每个通道（channels），负斜率参数是独立的，但在同一通道内的不同高度（height）和宽度（width）位置上，负斜率参数是共享的
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, kernel_size=(3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    bounding_box_regression = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([input], [classifier, bounding_box_regression])
    model.load_weights(weight_path, by_name=True)
    return model


def create_Rnet(weight_path):
    input = Input(shape=[24, 24, 3])
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    classifier = Dense(2, name='conv5-1', activation='softmax')(x)
    bounding_box_regression = Dense(4, name='conv5-2')(x)

    model = Model([input], [classifier, bounding_box_regression])
    model.load_weights(weight_path, by_name=True)

    return model


def create_Onet(weights_path):
    input = Input(shape=[48, 48, 3])
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)
    classifier = Dense(2, name='conv6-1', activation='softmax')(x)
    bounding_box_regression = Dense(4, name='conv6-2')(x)
    landmark_regression = Dense(10, name='conv6-3')(x)

    model = Model([input], [classifier, bounding_box_regression, landmark_regression])
    model.load_weights(weights_path, by_name=True)
    return model


class mtcnn:
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')

    def detectFace(self, img, threshold):
        # -----------------------------#
        #   归一化，加快收敛速度
        #   把[0,255]映射到(-1,1)
        # -----------------------------#
        copy_img = (img.copy() - 127.5) / 127.5
        origin_height, origin_width, _ = img.shape
        # -----------------------------#
        #   计算原始输入图像
        #   每一次缩放的比例
        # -----------------------------#
        scales = calculateScales(img)

        out = []
        # -----------------------------#
        #   粗略计算人脸框
        #   pnet部分
        # -----------------------------#
        for scale in scales:
            h = int(origin_height * scale)
            w = int(origin_width * scale)
            scale_img = cv.resize(copy_img, (w, h))
            input = scale_img.reshape(1, *scale_img.shape)
            output = self.Pnet.predict(input)
            out.append(output)

        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            # out = [
            #     [
            #         [cls_prob_0, cls_prob_1, ...],  # 第一个缩放级别的输出
            #         [roi_0, roi_1, ...]  # 第一个缩放级别的 ROI
            #     ],
            #     ...
            # ]
            # 有人脸的概率
            cls_prob = out[i][0][0][:, :, 1]
            # 其对应的框的位置
            roi = out[i][1][0]
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            print(cls_prob.shape)
            rectangle = detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_width, origin_height,
                                          threshold[0])
            rectangles.extend(rectangle)

            # 进行非极大抑制

        rectangles = NMS(rectangles, 0.7)

        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   稍微精确计算人脸框
        #   Rnet部分
        # -----------------------------#
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)
        out = self.Rnet.predict(predict_24_batch)

        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        rectangles = filter_face_24net(cls_prob, roi_prob, rectangles, origin_width, origin_height, threshold[1])

        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   计算人脸框
        #   onet部分
        # -----------------------------#
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        rectangles = filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_width, origin_height,
                                       threshold[2])

        return rectangles
