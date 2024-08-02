import colorsys
import copy
import os

import keras
from keras import backend as K, Input, Model
from PIL import Image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.engine import Layer
from keras.layers import Conv2D, Reshape, Flatten, TimeDistributed, Dense
#from util import anchors
from util import config
from nets.resnet import ResNet50, classifier_layers
from nets import RoiPoolingConv
from util import utils
def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height


class frcnn(object):
    _default = {
        'model_path': 'model_data/voc_weights.h5',
        'classes_path': 'model_data/voc_classes.txt',
        'confidence': 0.7
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._default:
            return cls._default[n]
        else:
            return "未知属性名:" + n

    def __init__(self, **kwargs):
        self.__dict__.update(self._default)
        self.class_names = self._get_class()
        #获取当前会话，保存在self.sess中，使得frcnn这个对象可以在后续
        #操作中利用这个会话来计算
        self.sess = K.get_session()
        self.config = config.Config()
        self.generate()
        self.bbox_util = utils.BBoxUtility()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            # input_length += 6
            filter_sizes = [7, 3, 1, 1]
            padding = [3, 1, 0, 0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
            return input_length

        return get_output_length(width), get_output_length(height)
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        self.num_classes = len(self.class_names) + 1
        self.rpn_model , self.class_model = get_predict_model(self.config, self.num_classes)
        self.rpn_model.load_weights(self.model_path, by_name=True) #by_name表示仅加载具有匹配名称的层的权重。
        self.class_model.load_weights(self.model_path, by_name=True,skip_mismatch=True)#skip_mismatch表示跳过那些名称匹配但形状不匹配的层的权重加载
        #每个元组的第一个数字（例如 0.0、0.05、0.1、0.15）代表的是色调（Hue），
        # 它决定了颜色的基本类型。色调是一个角度值，
        # 通常以 0 到 1 或者 0 到 360 度来表示，这里使用的是 0 到 1 的范围
        #0.0 或 1.0 对应于红色。
        #0.33 对应于绿色。
        #0.66 对应于蓝色。
        #中间值则代表这些基本颜色之间的过渡色。每组最后2个1代表亮度和饱和度为1
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        #*x代表解包hsv_tuples列表，map函数将一个个数据传入hsv_to_rgb执行
        self.colors= list(map(lambda x:colorsys.hsv_to_rgb(*x),hsv_tuples))
        self.colors = [(int(color_tuple[0]*255),int(color_tuple[1]*255),int(color_tuple[2]*255)) for color_tuple in self.colors]





def get_rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    x_class = Reshape((-1, 1), name="classification")(x_class)
    x_regr = Reshape((-1, 4), name="regression")(x_regr)
    return [x_class, x_regr, base_layers]

def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    out_roi_pool = RoiPoolingConv.RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

def get_predict_model(config, num_classes):
    input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None, None, 1024))
    base_layer = ResNet50(input)
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    #print(num_anchors)

    #建立RPN网络结构
    #kernel_initializer值为zeros，意味着卷积核的权重初始化全为0
    #uniform意味着初始化权重到一个小范围的均匀分布
    #normal意味着随机初始化权重为一个均值为0的小标准差正态分布
    #这种normal确保权重的均值为0，有助于避免梯度消失或者爆炸
    # x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
    #     base_layer)
    # x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='normal', name='rpn_out_class')(x)
    # #激活函数选择linear表示不会对卷积层输出做非线性变换，这个卷积用于预测回归值所以要选择线性激活函数
    # #权重矩阵设置为0意味着模型从0开始学习如何调整锚框到合理位置
    # x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    #
    # #第一个维度表示框的数量，第二个维度表示概率值
    # x_class = Reshape((-1, 1), name='classification')(x_class)
    # #第二个维度表示一个锚框有四个偏移量
    # x_regr = Reshape((-1, 4), name='regression')(x_regr)
    # rpn = [x_class, x_regr, base_layer]
    # rpn_model = Model(inputs=input, outputs=rpn)
    rpn = get_rpn(base_layer,num_anchors)
    rpn_model = Model(inputs=input, outputs=rpn[:2])

    #得到分类网络
    # pooling_regions = 14
    # input_shape = (config.num_rois, 14, 14, 1024)
    # #poi pooing卷积
    # #这里得到的out_roi_pool就是一个（1，32，14，14，1024）这样的结果
    # out_roi_pool = RoiPoolingConv.RoiPoolingConv(pool_size = pooling_regions,num_rois = config.num_rois)([base_layer, roi_input])
    # #得到out_roi_pool后要对这些数据做分类
    # #这里的classifier_layers没有接全链接，反而又做了resnet卷积，得到池化结果(1, 32, 1, 1, 2048)，trainable参数用于控制 Keras 层中的权重是否在模型训练过程中更新
    # out = classifier_layers(out_roi_pool,input_shape=input_shape,trainable=True)
    # #有TimeDistributed的话，实际上是在序列中的每个时间步独立执行的，我的序列长度为32
    # #所以拍扁操作实际上在1，1，2048上进行，也就是无关1和32，那么形状会变为(1,32,1*1*2048)
    # out = TimeDistributed(Flatten())(out)
    # #进入fc,out_class的输出形状是（1，32，21），out_regr的输出形状是(1,32,80)
    # out_class = TimeDistributed(Dense(num_classes,activation='softmax',kernel_initializer='zero'),name='dense_class')(out)
    # out_regr =TimeDistributed(Dense(4*(num_classes-1),activation='linear',kernel_initializer='zero'),name='dense_regress')(out)
    # classifier =  [out_class,out_regr]
    #
    # classifier_model = Model(inputs=[feature_map_input,roi_input],outputs=classifier)
    #return rpn_model, classifier_model
    classifier = get_classifier(base_layer,roi_input, config.num_rois,num_classes,trainable=True)
    classifier_model = Model([input,roi_input],classifier)
    return rpn_model, classifier_model







image = Image.open('img/street.jpg')
h, w, c = np.shape(image)
old_width = w
old_height = h
old_image = copy.deepcopy(image)
new_width, new_height = get_new_img_size(old_width, old_height)

image = image.resize((new_width, new_height))
photo = np.array(image, dtype=np.float64)

#给（600，600，3）的序号0位置上生成一个维度变成（1，600，600，3）
photo = np.expand_dims(photo, axis=0)
#print(photo)
photo = preprocess_input(photo)

frcnn = frcnn()
preds = frcnn.rpn_model.predict(photo)

image_output = frcnn.get_img_output_length(new_width, new_height)


def genreate_anchors(sizes=None,ratios=None):
    if sizes is None:
        sizes = config.Config().anchor_box_scales
    if ratios is None:
        ratios = config.Config().anchor_box_ratios

    num_anchors = len(sizes) * len(ratios)
    anchors = np.zeros((num_anchors, 4))

    #这里的anchors[:,2:]生成的结果就是后两列有值
    anchors[:,2:] = np.tile(sizes, (2, len(ratios))).T

    for i in range(len(ratios)):
        anchors[i*3:i*3+3,2] = anchors[i*3:i*3+3,2]*ratios[i][0]
        anchors[i*3:i*3+3,3] = anchors[i*3:i*3+3,3]*ratios[i][1]
    #np.tile 函数可以重复一个数组的元素来构建一个新的数组
    #第一个参数表示要重复的数组，第二个参数表示沿着2个轴重复的次数
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def shift(shape, anchors, stride=16):
    shift_x = (np.arange(0,shape[0],dtype=keras.backend.floatx())+0.5)* stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
    print(shift_x, shift_y)
    shift_x,shift_y = np.meshgrid(shift_x,shift_y)
    print(shift_x,shift_y)










anchors = genreate_anchors()
shift(image_output,anchors =anchors)
anchors = get_anchors(self.get_img_output_length(width, height), width, height)

rpn_results = self.bbox_util.detection_out(preds, anchors, 1, confidence_threshold=0)
# print(rpn_results)
# R = rpn_results[0][:,2:]
R = rpn_results[0][:, 2:]

R[:, 0] = np.array(np.round(R[:, 0] * width / self.config.rpn_stride), dtype=np.int32)
R[:, 1] = np.array(np.round(R[:, 1] * height / self.config.rpn_stride), dtype=np.int32)
R[:, 2] = np.array(np.round(R[:, 2] * width / self.config.rpn_stride), dtype=np.int32)
R[:, 3] = np.array(np.round(R[:, 3] * height / self.config.rpn_stride), dtype=np.int32)

R[:, 2] -= R[:, 0]
R[:, 3] -= R[:, 1]
base_layer = preds[2]

delete_line = []
for i, r in enumerate(R):
    if r[2] < 1 or r[3] < 1:
        delete_line.append(i)
R = np.delete(R, delete_line, axis=0)

bboxes = []
probs = []
labels = []
for jk in range(R.shape[0] // self.config.num_rois + 1):
    ROIs = np.expand_dims(R[self.config.num_rois * jk:self.config.num_rois * (jk + 1), :], axis=0)

    if ROIs.shape[1] == 0:
        break

    if jk == R.shape[0] // self.config.num_rois:
        # pad R
        curr_shape = ROIs.shape
        target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
        ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
        ROIs_padded[:, :curr_shape[1], :] = ROIs
        ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
        ROIs = ROIs_padded

    [P_cls, P_regr] = self.model_classifier.predict([base_layer, ROIs])

    for ii in range(P_cls.shape[1]):
        if np.max(P_cls[0, ii, :]) < self.confidence or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
            continue

        label = np.argmax(P_cls[0, ii, :])

        (x, y, w, h) = ROIs[0, ii, :]

        cls_num = np.argmax(P_cls[0, ii, :])

        (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
        tx /= self.config.classifier_regr_std[0]
        ty /= self.config.classifier_regr_std[1]
        tw /= self.config.classifier_regr_std[2]
        th /= self.config.classifier_regr_std[3]

        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h

        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

        x2 = cx1 + w1 / 2
        y2 = cy1 + h1 / 2

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        bboxes.append([x1, y1, x2, y2])
        probs.append(np.max(P_cls[0, ii, :]))
        labels.append(label)

if len(bboxes) == 0:
    return old_image

# 筛选出其中得分高于confidence的框
labels = np.array(labels)
probs = np.array(probs)
boxes = np.array(bboxes, dtype=np.float32)
boxes[:, 0] = boxes[:, 0] * self.config.rpn_stride / width
boxes[:, 1] = boxes[:, 1] * self.config.rpn_stride / height
boxes[:, 2] = boxes[:, 2] * self.config.rpn_stride / width
boxes[:, 3] = boxes[:, 3] * self.config.rpn_stride / height
results = np.array(
    self.bbox_util.nms_for_out(np.array(labels), np.array(probs), np.array(boxes), self.num_classes - 1, 0.4))

top_label_indices = results[:, 0]
top_conf = results[:, 1]
boxes = results[:, 2:]
boxes[:, 0] = boxes[:, 0] * old_width
boxes[:, 1] = boxes[:, 1] * old_height
boxes[:, 2] = boxes[:, 2] * old_width
boxes[:, 3] = boxes[:, 3] * old_height

font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // width
image = old_image
for i, c in enumerate(top_label_indices):
    predicted_class = self.class_names[int(c)]
    score = top_conf[i]

    left, top, right, bottom = boxes[i]
    top = top - 5
    left = left - 5
    bottom = bottom + 5
    right = right + 5

    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
    right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

    # 画框框
    label = '{} {:.2f}'.format(predicted_class, score)
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)
    label = label.encode('utf-8')
    print(label)

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i],
            outline=self.colors[int(c)])
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=self.colors[int(c)])
    draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
    del draw










