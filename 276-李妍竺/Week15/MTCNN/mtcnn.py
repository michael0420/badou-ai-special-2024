from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Activation, \
    Input, Reshape, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
import cv2
import utils

#-----------------------------#
#   粗略获取人脸框
#   输出bbox位置和是否有人脸
#-----------------------------#

def create_Pnet(weights_path):
    inputs = Input(shape=[None,None,3])

    x = Conv2D(10,(3,3),strides=1,name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)   #prelu:小于0的一段向第三象限轻微发展
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), strides=1, name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    #无激活函数，线性
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model(inputs, [classifier, bbox_regress])
    model.load_weights(weights_path, by_name=True)

    return model

#-----------------------------#
#   mtcnn的第二段
#   精修框
#-----------------------------#
def create_Rnet(weights_path):
    inputs = Input(shape=[24, 24, 3])

    # 24x24x3 -> 22x22x28 -> 11x11x28
    x = Conv2D(28, (3, 3), strides=1, name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # 11x11x28 -> 9x9x48 -> 4x4x48
    x = Conv2D(48, (3, 3), strides=1, name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)
    x = MaxPooling2D((3, 3), strides=2)(x)

    # 4x4x48 -> 3x3x64
    x = Conv2D(64, (2, 2), strides=1, name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    # 3x3x64 -> 64x3x3 (维度置换nhwc -> ncwh)
    x = Permute((3, 2, 1))(x)

    x = Flatten()(x)
    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='PReLU4')(x)

    # 128 -> 2      128 -> 4
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)

    model = Model(inputs, [classifier, bbox_regress])
    model.load_weights(weights_path, by_name=True)

    return model


#-----------------------------#
#   mtcnn的第三段
#   精修框并获得五个点
#-----------------------------#
def create_Onet(weights_path):
    inputs = Input(shape=[48, 48, 3])
    # 48x48x3 -> 46x46x32 -> 23x23x32
    x = Conv2D(32, (3, 3), strides=1, name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # 23x23x32 -> 21x21x64 -> 10x10x64
    x = Conv2D(64, (3, 3), strides=1, name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPooling2D((3, 3), strides=2)(x)

    # 10x10x64 -> 8x8x64 -> 4x4x64
    x = Conv2D(64, (3, 3), strides=1, name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPooling2D((2, 2))(x)

    # 4x4x64 -> 3x3x128
    x = Conv2D(128, (2, 2), strides=1, name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    # 3x3x128 -> 128x3x3
    x = Permute((3, 2, 1))(x)

    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 鉴别
    # 256 -> 2  256 -> 4  256 -> 10
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    bbox_regress = Dense(4, name='conv6-2')(x)
    landmark_regress = Dense(10, name='conv6-3')(x)

    model = Model(inputs, [classifier, bbox_regress, landmark_regress])
    model.load_weights(weights_path)

    return model

class mtcnn:
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')

    def detectFace(self,img,threshold):
        # -----------------------------#
        #   归一化，加快收敛速度
        #   把[0,255]映射到(-1,1)
        # -----------------------------#
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        # -----------------------------#
        #   计算原始输入图像
        #   每一次缩放的比例
        # -----------------------------#
        scales = utils.calculateScales(img)
        out = []

        # -----------------------------#
        #   粗略计算人脸框
        #   pnet部分
        # -----------------------------#
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)
            # 图像金字塔中的每张图片分别传入Pnet得到output
            output = self.Pnet.predict(inputs)
            out.append(output)
        img_num = len(scales)
        rectangles = []

        for i in range(img_num):
            # 有人脸的概率（i表示第i个output，第一个0表示是classifier，第二个0表示feature map
            # [:,:,1]表示第二个通道，这个通道的值表示正确的概率）
            # 从这个通道中提取值得到shape=(out_h,out_w)的矩阵
            cls_prob = out[i][0][0][:, :, 1]
            # 其对应的框的位置（i表示第i个output，1表示是bbox，0表示唯一一个feature map
            roi = out[i][1][0]

            # 取出每个缩放后图片的长宽
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            print(cls_prob.shape)

            # 解码过程
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        # 进行非极大值抑制
        rectangles = utils.nms(rectangles, 0.7)
        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   稍微精确计算人脸框
        #   Rnet部分
        # -----------------------------#
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
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   计算人脸框
        #   onet部分
        # -----------------------------#
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

        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles,origin_w, origin_h, threshold[2])

        return rectangles












