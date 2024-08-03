from nets.resnet import ResNet50, classifier_layers
from nets.RoiPoolingConv import RoiPoolingConv
from keras.layers import Conv2D, Input, TimeDistributed, Flatten, Dense, Reshape
from keras.models import Model


def get_rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    # activation='linear'这样写是因为regression后不需要加激活函数。

    x_class = Reshape((-1, 1), name="classification")(x_class)
    x_regr = Reshape((-1, 4), name="regression")(x_regr)
    return [x_class, x_regr, base_layers]


def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]


def get_model(config, num_classes):
    # 输入层定义，用于接收图像数据
    inputs = Input(shape=(None, None, 3))
    # 输入层定义，用于接收RoI（Region of Interest）数据
    roi_input = Input(shape=(None, 4))

    # 使用ResNet50作为基础模型
    base_layers = ResNet50(inputs)

    # 计算anchor的数量
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)

    # 构建RPN网络
    rpn = get_rpn(base_layers, num_anchors)
    # 创建仅包含RPN预测输出的模型
    model_rpn = Model(inputs, rpn[:2])

    # 构建分类器，用于对RoI进行分类和回归
    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    # 创建包含输入图像和RoI输入，以及分类器输出的模型
    model_classifier = Model([inputs, roi_input], classifier)

    # 创建一个包含所有输出的模型，即RPN和分类器的输出
    model_all = Model([inputs, roi_input], rpn[:2] + classifier)

    # 返回RPN模型、分类器模型和包含所有输出的综合模型
    return model_rpn, model_classifier, model_all



def get_predict_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None, None, 1024))

    base_layers = ResNet50(inputs)  # conv layer中的基础卷积结构
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)

    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn, model_classifier_only
