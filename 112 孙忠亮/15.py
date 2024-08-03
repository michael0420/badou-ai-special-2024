
from keras.layers import Conv2D, Input,MaxPool2D, Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model
import numpy as np
import cv2

class MtCnn():
    def __init__(self,pnet_path,rnet_path,onet_path):
        self.Pnet = self.create_Pnet(pnet_path)
        self.Rnet = self.create_Rnet(rnet_path)
        self.Onet = self.create_Onet(onet_path)
    def create_Pnet(self,weight_path):
        input = Input(shape=[None, None, 3])
        x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
        x = MaxPool2D(pool_size=2)(x)
        x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)
        x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)
        classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
        bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)
        model = Model([input], [classifier, bbox_regress])
        model.load_weights(weight_path, by_name=True)
        return model
    def create_Rnet(self,weight_path):
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
    def create_Onet(self,weight_path):
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
        classifier = Dense(2, activation='softmax', name='conv6-1')(x)
        bbox_regress = Dense(4, name='conv6-2')(x)
        landmark_regress = Dense(10, name='conv6-3')(x)
        model = Model([input], [classifier, bbox_regress, landmark_regress])
        model.load_weights(weight_path, by_name=True)
        return model
    def calculateScales(self,img):
        copy_img = img.copy()
        pr_scale = 1.0
        h, w, _ = copy_img.shape
        if min(w, h) > 500:
            pr_scale = 500.0 / min(h, w)
            w = int(w * pr_scale)
            h = int(h * pr_scale)
        elif max(w, h) < 500:
            pr_scale = 500.0 / max(h, w)
            w = int(w * pr_scale)
            h = int(h * pr_scale)
        scales = []
        factor = 0.709
        factor_count = 0
        minl = min(h, w)
        while minl >= 12:
            scales.append(pr_scale * pow(factor, factor_count))
            minl *= factor
            factor_count += 1
        return scales

    def detect_face_12net(self,cls_prob, roi, out_side, scale, width, height, threshold):
        cls_prob = np.swapaxes(cls_prob, 0, 1)
        roi = np.swapaxes(roi, 0, 2)
        stride = 0
        if out_side != 1:
            stride = float(2 * out_side - 1) / (out_side - 1)
        (x, y) = np.where(cls_prob >= threshold)

        boundingbox = np.array([x, y]).T
        bb1 = np.fix((stride * (boundingbox) + 0) * scale)
        bb2 = np.fix((stride * (boundingbox) + 11) * scale)
        boundingbox = np.concatenate((bb1, bb2), axis=1)
        dx1 = roi[0][x, y]
        dx2 = roi[1][x, y]
        dx3 = roi[2][x, y]
        dx4 = roi[3][x, y]
        score = np.array([cls_prob[x, y]]).T
        offset = np.array([dx1, dx2, dx3, dx4]).T
        boundingbox = boundingbox + offset * 12.0 * scale
        rectangles = np.concatenate((boundingbox, score), axis=1)
        rectangles = self.rect2square(rectangles)
        pick = []
        for i in range(len(rectangles)):
            x1 = int(max(0, rectangles[i][0]))
            y1 = int(max(0, rectangles[i][1]))
            x2 = int(min(width, rectangles[i][2]))
            y2 = int(min(height, rectangles[i][3]))
            sc = rectangles[i][4]
            if x2 > x1 and y2 > y1:
                pick.append([x1, y1, x2, y2, sc])
        return self.NMS(pick, 0.3)

    def rect2square(self,rectangles):
        w = rectangles[:, 2] - rectangles[:, 0]
        h = rectangles[:, 3] - rectangles[:, 1]
        l = np.maximum(w, h).T
        rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
        rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
        rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
        return rectangles

    def NMS(self,rectangles, threshold):
        if len(rectangles) == 0:
            return rectangles
        boxes = np.array(rectangles)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]
        area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
        I = np.array(s.argsort())
        pick = []
        while len(I) > 0:
            xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])  # I[-1] have hightest prob score, I[0:-1]->others
            yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
            xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
            yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
            pick.append(I[-1])
            I = I[np.where(o <= threshold)[0]]
        result_rectangle = boxes[pick].tolist()
        return result_rectangle

    def filter_face_24net(self,cls_prob, roi, rectangles, width, height, threshold):

        prob = cls_prob[:, 1]
        pick = np.where(prob >= threshold)
        rectangles = np.array(rectangles)

        x1 = rectangles[pick, 0]
        y1 = rectangles[pick, 1]
        x2 = rectangles[pick, 2]
        y2 = rectangles[pick, 3]

        sc = np.array([prob[pick]]).T

        dx1 = roi[pick, 0]
        dx2 = roi[pick, 1]
        dx3 = roi[pick, 2]
        dx4 = roi[pick, 3]

        w = x2 - x1
        h = y2 - y1

        x1 = np.array([(x1 + dx1 * w)[0]]).T
        y1 = np.array([(y1 + dx2 * h)[0]]).T
        x2 = np.array([(x2 + dx3 * w)[0]]).T
        y2 = np.array([(y2 + dx4 * h)[0]]).T

        rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
        rectangles = self.rect2square(rectangles)
        pick = []
        for i in range(len(rectangles)):
            x1 = int(max(0, rectangles[i][0]))
            y1 = int(max(0, rectangles[i][1]))
            x2 = int(min(width, rectangles[i][2]))
            y2 = int(min(height, rectangles[i][3]))
            sc = rectangles[i][4]
            if x2 > x1 and y2 > y1:
                pick.append([x1, y1, x2, y2, sc])
        return self.NMS(pick, 0.3)

    def filter_face_48net(self,cls_prob, roi, pts, rectangles, width, height, threshold):

        prob = cls_prob[:, 1]
        pick = np.where(prob >= threshold)
        rectangles = np.array(rectangles)

        x1 = rectangles[pick, 0]
        y1 = rectangles[pick, 1]
        x2 = rectangles[pick, 2]
        y2 = rectangles[pick, 3]

        sc = np.array([prob[pick]]).T

        dx1 = roi[pick, 0]
        dx2 = roi[pick, 1]
        dx3 = roi[pick, 2]
        dx4 = roi[pick, 3]

        w = x2 - x1
        h = y2 - y1

        pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
        pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
        pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
        pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
        pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
        pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
        pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
        pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
        pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
        pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T

        x1 = np.array([(x1 + dx1 * w)[0]]).T
        y1 = np.array([(y1 + dx2 * h)[0]]).T
        x2 = np.array([(x2 + dx3 * w)[0]]).T
        y2 = np.array([(y2 + dx4 * h)[0]]).T

        rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                    axis=1)

        pick = []
        for i in range(len(rectangles)):
            x1 = int(max(0, rectangles[i][0]))
            y1 = int(max(0, rectangles[i][1]))
            x2 = int(min(width, rectangles[i][2]))
            y2 = int(min(height, rectangles[i][3]))
            if x2 > x1 and y2 > y1:
                pick.append([x1, y1, x2, y2, rectangles[i][4],
                             rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9],
                             rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13],
                             rectangles[i][14]])
        return self.NMS(pick, 0.3)

    def detectFace(self, img, threshold):
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        scales = self.calculateScales(img)
        out = []
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
            cls_prob = out[i][0][0][:,:,1]
            roi = out[i][1][0]
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            print(cls_prob.shape)
            rectangle = self.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)
        rectangles = self.NMS(rectangles, 0.7)
        if len(rectangles) == 0:
            return rectangles
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
        rectangles = self.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        if len(rectangles) == 0:
            return rectangles
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
        rectangles = self.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        return rectangles

if __name__=='__main__':
    img = cv2.imread('img/test1.jpg')
    pnet_path='model_data/pnet.h5'
    rnet_path='model_data/rnet.h5'
    onet_path='model_data/onet.h5'
    model = MtCnn(pnet_path,rnet_path,onet_path)
    threshold = [0.5,0.6,0.7]  # 三段网络的置信度阈值不同
    rectangles = model.detectFace(img, threshold)
    print(rectangles)
    draw = img.copy()

    for rectangle in rectangles:
        if rectangle is not None:
            W = -int(rectangle[0]) + int(rectangle[2])
            H = -int(rectangle[1]) + int(rectangle[3])
            paddingH = 0.01 * W
            paddingW = 0.02 * H
            crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
            if crop_img is None:
                continue
            if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                continue
            cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)

            for i in range(5, 15, 2):
                cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

    cv2.imwrite("img/out.jpg",draw)

    cv2.imshow("test", draw)
    c = cv2.waitKey(0)




'''..........................................................................'''
import tensorflow as tf

class yoloV3:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train


    def batch_normalization_layer(self, input_layer, name = None, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        bn_layer = tf.layers.batch_normalization(inputs = input_layer,
            momentum = norm_decay, epsilon = norm_epsilon, center = True,
            scale = True, training = training, name = name)
        return tf.nn.leaky_relu(bn_layer, alpha = 0.1)

    def conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias = False, strides = 1):
        conv = tf.layers.conv2d(
            inputs = inputs, filters = filters_num,
            kernel_size = kernel_size, strides = [strides, strides], kernel_initializer = tf.glorot_uniform_initializer(),
            padding = ('SAME' if strides == 1 else 'VALID'), kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 5e-4), use_bias = use_bias, name = name)
        return conv
    def Residual_block(self, inputs, filters_num, blocks_num, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self.conv2d_layer(inputs, filters_num, kernel_size = 3, strides = 2, name = "conv2d_" + str(conv_index))
        layer = self.batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self.conv2d_layer(layer, filters_num // 2, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            layer = self.batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer = self.conv2d_layer(layer, filters_num, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            layer = self.batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    def darkNet53(self, inputs, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        with tf.variable_scope('darknet53'):
            conv = self.conv2d_layer(inputs, filters_num = 32, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            conv, conv_index = self.Residual_block(conv, conv_index = conv_index, filters_num = 64, blocks_num = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self.Residual_block(conv, conv_index = conv_index, filters_num = 128, blocks_num = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self.Residual_block(conv, conv_index = conv_index, filters_num = 256, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route1 = conv
            conv, conv_index = self.Residual_block(conv, conv_index = conv_index, filters_num = 512, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route2 = conv
            conv, conv_index = self.Residual_block(conv, conv_index = conv_index,  filters_num = 1024, blocks_num = 4, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        return  route1, route2, conv, conv_index

    def yolo_block(self, inputs, filters_num, out_filters, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        conv = self.conv2d_layer(inputs, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        route = conv
        conv = self.conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters_num = out_filters, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index), use_bias = True)
        conv_index += 1
        return route, conv, conv_index

    def yolo_net(self, inputs, num_anchors, num_classes, training = True):
        conv_index = 1
        conv2d_26, conv2d_43, conv, conv_index = self.darkNet53(inputs, conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
        with tf.variable_scope('yolo'):
            conv2d_57, conv2d_59, conv_index = self.yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv2d_60 = self.conv2d_layer(conv2d_57, filters_num = 256, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_60 = self.batch_normalization_layer(conv2d_60, name = "batch_normalization_" + str(conv_index),training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]], name='upSample_0')
            route0 = tf.concat([unSample_0, conv2d_43], axis = -1, name = 'route_0')
            conv2d_65, conv2d_67, conv_index = self.yolo_block(route0, 256, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv2d_68 = self.conv2d_layer(conv2d_65, filters_num = 128, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_68 = self.batch_normalization_layer(conv2d_68, name = "batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upSample_1')
            route1 = tf.concat([unSample_1, conv2d_26], axis = -1, name = 'route_1')
            _, conv2d_75, _ = self.yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]



'''............................................................................................'''

import os
import colorsys
import numpy as np
import tensorflow as tf
from yoloV3_model import yoloV3


class yolo_predictor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.classes_path = classes_file
        self.anchors_path = anchors_file
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)  # 将相对路径扩展为完整的绝对路径。
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]  # 移除字符串首位空格
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        # 获得特征
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        # 寻找在原图上的位置
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        # 获得置信度
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    # 获得在原图上框的位置
    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        # 416,416
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        # 实际图片的大小
        image_shape = tf.cast(image_shape, dtype=tf.float32)

        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    def _get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        # 这里构建13*13*1*2的矩阵
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)

        # x,y坐标归一化
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # w,h归一化
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        # 每一个特征层对应三个先验框
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = tf.shape(yolo_outputs[0])[1: 3] * 32
        # 对三个特征层的输出获取每个预测box坐标和box的分数
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                        len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        # 放在一行
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []
        # 对每一个类进行判断
        for c in range(len(self.class_names)):
            # 取出所有类为c的box
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有类为c的分数
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            # 非极大抑制
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                     iou_threshold=self.nms_threshold)

            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)
        return boxes_, scores_, classes_

    def predict(self, inputs, image_shape,norm_epsilon=1e-3,norm_decay=0.99,num_anchors=9,num_classes=80):

        model = yoloV3(norm_epsilon, norm_decay, self.anchors_path, self.classes_path, pre_train=False)
        # yolo_inference用于获得网络的预测结果
        output = model.yolo_net(inputs, num_anchors // 3, num_classes, training=False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)
        return boxes, scores, classes