import numpy as np
from keras.layers import Input, Conv2D, PReLU, MaxPool2D, Permute, Flatten, Dense
from keras.models import Model
from utils_homework import calculateScales
import cv2
import utils_homework as utils

def create_Pnet(weight_path):
    """
    第一段网络P-net
    :param weight_path:  训练好的模型参数文件路径
    :return:  model: classifier为是否有人脸的分类结果 -> (1, 1, 2)
                     bbox_regress为人脸框的示框左上角的横坐标的相对偏移，框左上角的纵坐标的相对偏移、框的宽度的误差、框的高度的误差 -> (1, 1, 4)
    """
    inputs = Input(shape=[None, None, 3])

    x = Conv2D(filters=10, kernel_size=(3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(filters=2, kernel_size=(1, 1), activation='softmax', name='conv4-1')(x)

    bbox_regress = Conv2D(filters=4, kernel_size=(1, 1), name='conv4-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

def create_Rnet(weight_path):
    """
    第二段网络R-net
    :param weight_path:  训练好的模型参数文件路径
    :return:  model: classifier为是否有人脸的分类结果 -> (1, 1, 2)
                     bbox_regress为人脸框的示框左上角的横坐标的相对偏移，框左上角的纵坐标的相对偏移、框的宽度的误差、框的高度的误差 -> (1, 1, 4)
    """
    inputs = Input(shape=[24, 24, 3])

    x = Conv2D(filters=28, kernel_size=(3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = Conv2D(filters=48, kernel_size=(3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(filters=64, kernel_size=(2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)
    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

def create_Onet(weight_path):
    """
    第三段网络O-net
    :param weight_path:  训练好的模型参数文件路径
    :return:  model: classifier为是否有人脸的分类结果 -> (1, 1, 2)
                     bbox_regress为人脸框的示框左上角的横坐标的相对偏移，框左上角的纵坐标的相对偏移、框的宽度的误差、框的高度的误差 -> (1, 1, 4)
                     landmark_regress为人脸五个特征（左眼、右眼、鼻子、左嘴角、右嘴角）的XY坐标 -> (1, 1, 10)
    """
    inputs = Input(shape=[48, 48, 3])

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(filters=128, kernel_size=(2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    bbox_regress = Dense(4, name='conv6-2')(x)
    landmark_regress = Dense(10, name='conv6-3')(x)

    model = Model([inputs], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model

class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')

    def detectFace(self, img, threshold):
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        scales = calculateScales(img)
        out = []

        # Pnet部分
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)

            output = self.Pnet.predict(inputs)
            out.append(output)

        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            # 有人脸的概率
            cls_prob = out[i][0][0][:, :, 1]
            # 对应的框的位置
            roi = out[i][1][0]

            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            print(cls_prob.shape)

            # 解码
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        rectangles = utils.NMS(rectangles, 0.7)
        if len(rectangles) == 0:
            return rectangles

        # Rnet部分
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

        # Onet部分
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

        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles
