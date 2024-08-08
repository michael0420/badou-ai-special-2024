from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
import utils_demo as utils
import cv2

'''
粗略获取人脸框
输出bbox位置是否有人脸
'''
def create_pnet(weight_path):
    input = Input(shape=[None,None,3])
    x = Conv2D(10,(3,3),strides=1,padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='PRelu1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16,(3,3),strides=1,padding='valid',name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu2')(x)

    x = Conv2D(32,(3,3),strides=1,padding='valid',name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu3')(x)

    classifier = Conv2D(2,(1,1),activation='softmax',name='conv4_1')(x)
    # 无激活函数，线性4个偏置
    bbox_regress = Conv2D(4,(1,1),name='conv4_2')(x)
    model = Model([input],[classifier,bbox_regress])
    model.load_weights(weight_path,by_name=True) # 按照层的名称而不是层的顺序来加载权重
    return model

'''
精修框
'''
def create_Rnet(weight_path):
    # 全连接层固定尺寸输入
    input = Input(shape=[24,24,3])
    x = Conv2D(28,(3,3),strides=1,padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='PRelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2,padding='same')(x) # padding='same' 保持输入特征图的空间尺寸

    x = Conv2D(48,(3,3),strides=1,padding='valid',name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu2')(x)
    x = MaxPool2D(pool_size=3,strides=2)(x) # padding='valid' 逐步减少特征图尺寸

    x = Conv2D(64,(2,2),strides=1,padding='valid',name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu3')(x)


    # 1. 调整维度和展平
    x = Permute((3,2,1))(x) # 对张量对维度重新排列
    x = Flatten()(x) # 展平

    # 2. 全连接层
    x = Dense(128,name='conv4')(x)
    x = PReLU(name='PRelu4')(x)

    # 3. 输出层
    classifier = Dense(2,activation='softmax',name='conv5-1')(x)
    bbox_regress = Dense(4,name='conv5-2')(x)

    # 4. 加载模型
    model = Model([input],[classifier,bbox_regress])
    model.load_weights(weight_path,by_name=True)
    return model

'''
精修框并获得五个点
'''
def create_Onet(weight_path):
    input = Input(shape=[48,48,3])

    x = Conv2D(32,(3,3),strides=1,padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='PRelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2,padding='same')(x)

    x = Conv2D(64,(3,3),strides=1,padding='valid',name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu2')(x)
    x = MaxPool2D(pool_size=3,strides=2)(x)

    x = Conv2D(64,(3,3),strides=1,padding='valid',name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(128,(2,2),strides=1,padding='valid',name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu4')(x)

    x = Permute((3,2,1))(x)
    x = Flatten()(x)

    # 全连接层
    x = Dense(256,name='conv5')(x)
    x = PReLU(name='PRelu5')(x)

    # 输出层
    classifier = Dense(2,activation='softmax',name='conv6-1')(x)
    bbox_regress = Dense(4,name='conv6-2')(x)
    landmark_regress = Dense(10,name='conv6-3')(x)

    model = Model([input],[classifier,bbox_regress,landmark_regress])
    model.load_weights(weight_path,by_name=True)
    return model

class mtcnn():
    def __init__(self):
        # 加载预训练模型
        self.Pnet = create_pnet(weight_path='model_data/pnet.h5')
        self.Rnet = create_Rnet(weight_path='model_data/rnet.h5')
        self.Onet = create_Onet(weight_path='model_data/onet.h5')

    def detectFace(self,img,threshold):
        # threshold:阈值参数

        # 1.归一化,加快收敛速度
        copy_img = (img.copy() -127.5) / 127.5
        origin_h,origin_w,_ = copy_img.shape

        # 2. 计算图像金字塔的缩放比例
        scales = utils.calculateScales(img) # 生成不同尺寸的图像

        # 3. P-Net
        out = []
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img,(ws,hs))
            inputs = scale_img.reshape(1, *scale_img.shape) # 调整为（1,h,w,c)
            output = self.Pnet.predict(inputs) # 图像金字塔输入P-Net
            out.append(output)
        # 4. 解码P-Net的输出
        image_num = len(scales) # 图像金字塔的图像数
        rectangles = [] # 储存候选人脸框
        for i in range(image_num):
            cls_prob = out[i][0][0][:,:,1] # 分类概率 是否有人脸的概率,[:,:,1]:选择类别为1的概率
            roi = out[i][1][0] # 提取边界回归框的输出
            out_h,out_w = cls_prob.shape # 获取分类概率的高度和宽度
            out_side = max(out_h, out_w) #获取分类概率图的最大边长
            # 得到人脸框
            rectangle = rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle) # 添加框
        # 5. 非极大值抑制
        rectangles = utils.NMS(rectangles,0.7) # 筛选出置信度大于0.7的框，去除重复的检测框

        # 6. R-Net精细检测，精细框
        predict_24_batch = [] # 24是网络输入尺寸
        for rectangle in rectangles:
            # 在原图上裁剪出框中的图像
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img,(24,24)) # 可以输入网络的尺寸
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch) #转换为矩阵
        out = self.Rnet.predict(predict_24_batch)

        cls_prob = out[0] # 分类
        roi_prob = out[1] # 边界回归框
        # 调整候选框位置
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

        if len(rectangles) == 0:
            return rectangles # 没有候选框

        # 7. O-Net 画人脸框，定位五官
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img,(48,48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)

        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]
        # 人脸画框 五官
        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob,rectangles, origin_w, origin_h, threshold[2])
        return rectangles






