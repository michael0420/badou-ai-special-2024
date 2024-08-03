from keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model
import numpy as np
import utils
import cv2


# 获取人脸框,输出bbox位置和是否有人脸
def create_Pnet(weight_path):
    input = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 无激活函数，线性
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model


#   精确框的位置
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
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model


#   获得框的五个点
def create_Onet(weight_path):

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

    # 鉴别
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    bbox_regress = Dense(4, name='conv6-2')(x)
    landmark_regress = Dense(10, name='conv6-3')(x)

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model


class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet('../../../../data/mtcnn-keras-master/model_data/pnet.h5')
        self.Rnet = create_Rnet('../../../../data/mtcnn-keras-master/model_data/rnet.h5')
        self.Onet = create_Onet('../../../../data/mtcnn-keras-master/model_data/onet.h5')

    def detectFace(self, img, threshold):

        #   归一化
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        # 计算原始输入图像,每一次缩放的比例
        scales = utils.calculateScales(img)
        out = []

        #   粗略计算人脸框
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)
            # 图像金字塔中的每张图片分别传入Pnet得到output
            output = self.Pnet.predict(inputs)
            # 将所有output加入out
            out.append(output)

        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            # 有人脸的概率
            cls_prob = out[i][0][0][:, :, 1]
            # 其对应的框的位置
            roi = out[i][1][0]
            # 取出每个缩放后图片的长宽
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            # print(cls_prob.shape)
            # 解码
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h,
                                                threshold[0])
            rectangles.extend(rectangle)

        # 非极大抑制
        rectangles = utils.NMS(rectangles, 0.7)
        if len(rectangles) == 0:
            return rectangles

        # 稍微精确计算人脸框
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

        # 计算人脸框
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
