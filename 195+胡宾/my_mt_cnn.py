from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
import utils
import cv2


# P-Net mtcnn 第一阶段
def create_p_net(weight_path):
    input = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='PreLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PreLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PreLU3')(x)

    classification = Conv2D(2, (1, 1), activation='softmax', padding='valid', name='conv4')(x)

    regression = Conv2D(4, (1, 1), name='conv5')(x)

    model = Model([input], [classification, regression])

    model.load_weights(weight_path, by_name=True)
    return model


# R-Net 第二阶段
def create_r_net(weight_path):
    input = Input(shape=[24, 24, 3])
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='RreLU1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='RreLU2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='RreLU3')(x)

    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    x = Dense(128, name='rconv4')(x)
    x = PReLU(name='RreLU4')(x)

    classification = Dense(2, activation='softmax', name='conv5-1')(x)

    regression = Dense(4, name='conv5-2')(x)

    model = Model([input], [classification, regression])
    model.load_weights(weight_path, by_name=True)
    return model

# O-Net 第三阶段
def create_Onet(weight_path):
    input = Input(shape = [48,48,3])
    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)
    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)
    # 3,3,128 -> 128,12,12
    x = Permute((3,2,1))(x)

    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5') (x)
    x = PReLU(name='prelu5')(x)

    # 鉴别
    # 256 -> 2 256 -> 4 256 -> 10
    classifier = Dense(2, activation='softmax',name='conv6-1')(x)
    bbox_regress = Dense(4,name='conv6-2')(x)
    landmark_regress = Dense(10,name='conv6-3')(x)

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model


class my_mt_cnn():
    def __init__(self):
        self.Pnet = create_p_net('model_data/pnet.h5')
        self.Rnet = create_r_net('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')

    def detectFace(self, img, threshold):
        # 归一化
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape

        scales = utils.calculateScales(img)

        out = []
        for scale in scales:
            hs = int(origin_h*scale)
            ws = int(origin_w*scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)
            output = self.Pnet.predict(inputs)
            out.append(output)

        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            # 有人脸的概率
            prob = out[i][0][0][:, :, 1]
            # 对应框的位置
            box = out[i][1][0]

            out_h, out_w = prob.shape
            out_side = max(out_h, out_w)
            print(prob.shape)

            rectangle = utils.detect_face_12net(prob, box, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        # 进行非极大值抑制
        rectangles = utils.NMS(rectangles, 0.7)
        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   稍微精确计算人脸框
        #   Rnet部分
        # -----------------------------#
        predict_rnet_batch = []
        for rect in rectangles:
            c_img = copy_img[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]
            resize_rnet_img = cv2.resize(c_img, (24, 24))
            predict_rnet_batch.append(resize_rnet_img)

        predict_rnet_batch = np.array(predict_rnet_batch)
        out = self.Rnet.predict(predict_rnet_batch)

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
        for rectang in rectangles:
            crop_img = copy_img[int(rectang[1]):int(rectang[3]), int(rectang[0]):int(rectang[2])]
            resize_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(resize_img)

        predict_batch = np.array(predict_batch)
        out_put = self.Onet.predict(predict_batch)
        cls_prob = out_put[0]
        pri_prob = out_put[1]
        pts_prob = out_put[2]

        rectangles = utils.filter_face_48net(cls_prob, pri_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles








