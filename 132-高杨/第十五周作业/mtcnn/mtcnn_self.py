import numpy as np
from keras.layers import Conv2D,Input,Dense,MaxPooling2D,Reshape,Activation,Flatten,Permute
from  keras.layers.advanced_activations import PReLU
from keras.models import Model,Sequential
import tensorflow as tf
import utils
import cv2


def create_Pnet(weight_path):

    input = Input(shape=[None,None,3])

    x =Conv2D(10,(3,3),strides=1,padding='valid')(input)
    x = PReLU(shared_axes=[1,2], name='Prelu1')(x)
    x = MaxPooling2D(padding=2)(x)

    x = Conv2D(16,(3,3),strides=1,padding='valid')(x)
    x = PReLU(shared_axes=[1,2],name='Prelu2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid')(x)
    x = PReLU(shared_axes=[1, 2], name='Prelu3')(x)

    classifier = Conv2D(2,(1,1),activation='softmax')(x)
    # 框无激活函数，线性
    bbox_regress = Conv2D(4,(1,1))(x)

    model = Model([input] , [classifier,bbox_regress])
    model.load_weights(weight_path,by_name=True)

    return model

# 第二极端 精修框

def create_Rnet(weight_path):
    input = Input(shape=[24,24,3])


    x =Conv2D(28,(3,3),strides=1,padding='valid')(input)
    x = PReLU(shared_axes=[1,2], name='RRelu1')(x)
    x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)

    # 11*11*28 -》 4，4，48  11 -1 // 2 + 3
    #新的高度 = ((原高度 - 卷积核高度 + 2 * 填充高度) / 步长高度) + 1
    x =Conv2D(48,(3,3),strides=1,padding='valid')(x)
    x = PReLU(shared_axes=[1,2], name='RRelu2')(x)
    x = MaxPooling2D(pool_size=3,strides=2)(x)

    # 4 4 48 3 3 64
    x =Conv2D(64,(2,2),strides=1,padding='valid')(x)
    x = PReLU(shared_axes=[1,2],name='RRelu3')(x)

    # 3 3 64  64 3  3
    x = Permute((3,2,1))(x)
    x = Flatten()(x)

    # 576  128
    x = Dense(128)(x)
    x = PReLU(name='RRelu4')(x)

    # 128  2 128 4
    classifier = Dense(2,activation='softmax')(x)
    bbox_regress = Dense(4,name='bbox_classifer')(x)

    model = Model([input],[classifier,bbox_regress])
    model.load_weights(weight_path,by_name=True)

    return model

def create_Onet(weight_path):
    input = Input(shape=[48,48,3])

    # 48 48 3  23 23 32
    x = Conv2D(32,(3,3),strides=1,padding='valid')(input)
    x = PReLU(shared_axes=[1,2],name='ORelu1')(x)
    x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)

    #23 23 32  10 10 64
    x = Conv2D(64, (3, 3), strides=1, padding='valid')(x)
    x = PReLU(shared_axes=[1, 2], name='ORelu2')(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # 10 1
    x = Conv2D(64, (3, 3), strides=1, padding='valid')(x)
    x = PReLU(shared_axes=[1, 2], name='ORelu3')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)


    # 4 4 64 3 3 128
    x = Conv2D(128, (2, 2), strides=1, padding='valid')(x)
    x = PReLU(shared_axes=[1, 2], name='ORelu4')(x)

   # 128 3 3
    x = Permute((3,2,1))(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x =PReLU()(x)

    classifer = Dense(2,activation='softmax')(x)
    bbox_regress = Dense(4)(x)
    landmark_regress =  Dense(10)(x)

    model = Model([input],[classifer,bbox_regress,landmark_regress])
    model.load_weights(weight_path,by_name=True)

    return model

class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Pnet('model_data/rnet.h5')
        self.Onet = create_Pnet('model_data/onet.h5')

    def detectFace(self,img,threshold):
        #  对图片进行归一化 加快收敛速度  【0 255】  （-1，1）
        copy_img = (img.copy() - 127.5)  / 127.5
        orgin_h,orgin_w = copy_img.shape

        # 计算原始输入图像 ，以及每一次缩放1比例
        scales = utils.calculateScales(img)

        out = []

        # 粗略计算人脸框 pnet部分
        for scale in scales:
            hs = int(orgin_h * scale)
            ws = int(orgin_w * scale)
            scale_img = cv2.resize(img,(ws,hs))
            #*scale_img.shape 是Python中的解包操作符，它将元组解包为其单独的元素
            inputs = scale_img.reshape(1,*scale_img)

            # 图像金字塔的每个图片传入Pnet 得到output
            output = self.Pnet.predict(inputs)
            # 讲所有output加入
            out.append(output)

        img_num = len(scales)

        rectangles = []
        for i in range(img_num):
            # 有人脸的概率
            cls_prob = out[i][0][0][:,:,1]

            # 对应框的位置
            roi = out[i][1][0]

            # 取出每个缩放后图片长宽
            out_h,out_w = cls_prob.shape
            out_side = max(out_h,out_w)
            print(cls_prob.shape)

            # 解码过程
            rectangle = utils.detect_face_12net(cls_prob,roi,out_side,1 / scales[i],orgin_w,orgin_h,threshold[0])
            rectangles.extend(rectangle)


        # 进行非极大值抑制

        rectangles = utils.NMS(rectangles,0.7)

        if len(rectangles) == 0:
            return rectangles




        #   第二步 稍微精确计算人脸框
        predict_24_batch =[]

        for rectangle in rectangles:
            copy_img =copy_img[int(rectangle[1]):int(rectangle[3]),int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(copy_img,(24,24))
            predict_24_batch.append(scale_img)


        predict_24_batch = np.array(predict_24_batch)
        out = self.Rnet.predict(predict_24_batch)

        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        rectangles = utils.filter_face_24net(cls_prob,roi_prob,rectangles,orgin_w,orgin_h,threshold[1])

        if len(rectangles) == 0:
            return rectangles


        # 计算人脸框 onet部分

        predict_batch = []
        for rectangle in rectangles:
            copy_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(copy_img, (48, 48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        cls_prob =output[0]
        roi_prob =output[1]
        pts_prob = output[2]

        rectangles = utils.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,orgin_w,orgin_h,threshold=threshold[2])

        return rectangles









