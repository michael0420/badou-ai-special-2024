# coding = utf-8
'''
    实现 mtcnn 人脸识别
'''

import cv2
import numpy as np
import tensorflow as tf
import utils
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers import Conv2D, Input, MaxPooling2D, Reshape, Activation, Flatten, Dense, Permute


def create_Pnet(path):
    # Pnet部分
    # 粗略获取人脸框，输出bbox位置和判断是否有人脸
    x_input = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(x_input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    bbox = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([x_input], [classifier, bbox])
    model.load_weights(path, by_name=True)

    return model

def create_Rnet(path):
    # Rnet部分
    # 精修框
    x_input = Input(shape=[24, 24, 3])
    # 24,24,3 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(x_input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # 11,11,28 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='PReLU4')(x)

    # 128 -> 2 128 -> 4
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox = Dense(4, name='conv5-2')(x)
    model = Model([x_input], [classifier, bbox])
    model.load_weights(path, by_name=True)

    return model

def create_Onet(path):
    # Onet部分
    # 精修框， 精确定位5个点
    x_input = Input(shape=[48, 48, 3])

    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(x_input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)
    x = MaxPooling2D(pool_size=2)(x)

    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (3, 3), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU4')(x)

    # 3,3,128 -> 128,12,12
    x = Permute((3, 2, 1))(x)

    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='PReLU5')(x)

    # 鉴别
    # 256 -> 2 256 -> 4 256 -> 10
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    bbox = Dense(4, name='conv6-2')(x)
    landmark = Dense(10, name='conv6-3')(x)

    model = Model([x_input, [classifier, bbox, landmark]])
    model.load_weights(path, by_name=True)

    return model

class mtcnn():

    def __init__(self):
        self.Pnet = create_Pnet(r'./model_data/pnet.h5')
        self.Rnet = create_Rnet(r'./model_data/rnet.h5')
        self.Onet = create_Onet(r'./model_data/onet.h5')

    def predictFace(self, img, threshold):
        # 归一化处理
        copy_img = (img.copy() - 127.5) / 127.5
        ori_h, ori_w, _ = copy_img.shape

        # 计算原始输入图像每次缩放的比例
        scales = utils.calculateScales(img)
        out = []

        # pnet 获取 人脸box
        for scale in scales:
            hs = int(ori_h * scale)
            ws = int(ori_w * scale)

            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)

            # 图像金字塔中的每张图片分别传入Pnet得到output
            output = self.Pnet.predict(inputs)
            out.append(output)

        img_num = len(scales)
        rectangles = []
        for i in range(img_num):
            # 有人脸的概率
            cls_prob = out[i][0][0][:, :, 1]
            # 其对应的框的位置
            roi = out[i][1][0]

            # 取出每个缩放后图片的长宽
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            print(cls_prob.shape)
            # 解码过程
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], ori_w, ori_h, threshold[0])
            rectangles.extend(rectangle)

        # nms非极大抑制
        rectangles = utils.NMS(rectangles, 0.7)

        if len(rectangles) == 0:
            return rectangles

        # rnet 人脸box校准
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)
        out = self.Rnet.predict(predict_24_batch)

        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, ori_w, ori_h, threshold[1])

        if len(rectangles) == 0:
            return rectangles

        # 0net 人脸框计算
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, ori_w, ori_h, threshold[2])

        return rectangles
