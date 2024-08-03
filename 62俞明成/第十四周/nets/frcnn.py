from nets.resnet import ResNet50, classifier_layers
from keras.layers import Conv2D, Input, TimeDistributed, Flatten, Dense, Reshape
from keras.models import Model
from nets.RoiPoolingConv import RoiPoolingConv


def get_rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    # 精简softmax，使用sigmoid完成二分类问题
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    # kernel_initializer='zero'卷积核权重初始化为0
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    x_class = Reshape((-1, 1), name="classification")(x_class)
    x_regr = Reshape((-1, 4), name="regression")(x_regr)
    return [x_class, x_regr, base_layers]


def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    # 类实例可以像函数一样被调用，这会触发该类的__call__方法。
    # 在 Keras 中，所有层都继承自 Layer 类，而 Layer 类定义了 __call__ 方法。
    # __call__方法的主要步骤如下：
    # 输入验证和处理：检查输入的格式和有效性，并将其转换为适当的张量格式。
    # 调用build方法（如果尚未构建）：如果这是第一次调用该层，会调用build方法以创建层的权重。
    # 调用call方法：传递输入数据给层的call方法，实际的计算在这里完成。
    # 返回输出：获取call方法的输出，并进行任何必要的后处理，然后返回最终结果
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]


def get_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    base_layers = ResNet50(inputs)

    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn[:2])

    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)

    model_all = Model([inputs, roi_input], rpn[:2] + classifier)
    return model_rpn, model_classifier, model_all


def get_predict_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None, None, 1024))

    base_layers = ResNet50(inputs)
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)  # 9
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)

    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn, model_classifier_only
