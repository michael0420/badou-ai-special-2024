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
