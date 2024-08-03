from resnet import ResNet50,classifier_layers
from keras.layers import Conv2D,Input,TimeDistributed,Flatten,Dense,Reshape
from keras.models import Model
from RoiPoolingConv import RoiPoolingConv


# 用于生成成region proposals
def get_rpn(base_layers, num_anchors):
    # 这一层接受base_layer的输出作为输入 卷积核3*3 通道数512 补零 激活函数relu 卷积核初始方式为正态分布
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    # 这一层定义一个分类层 接受x的输出为输入 卷积核大小1*1 个数为anchors的个数 卷积核初始化为uniform 激活函数使用sigmoid
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    # 这一层定义一个回归层 接受x的输出为输入 卷积核大小1*1 个数为四倍的anchors的个数 因为有四个图像信息返回 卷积核初始化为uniform 激活函数使用线性
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    # reshape
    x_class = Reshape((-1,1),name="classification")(x_class)
    x_regr = Reshape((-1,4),name="regression")(x_regr)
    return [x_class, x_regr, base_layers]


def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    # RoiPoolingConv 处理 ROI 池化
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    # classifier_layers 可能包含多个层 用于进一步处理 ROI 池化后的特征
    out = classifier_layers(out_roi_pool, trainable=trainable)  # 注意：这里假设 classifier_layers 能处理 trainable 参数
    # 将每个 ROI 的特征图展平
    out = TimeDistributed(Flatten())(out)
    # 类别预测层
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='he_normal'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # 边界框回归层
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='he_normal'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

def get_model(config,num_classes):
    inputs = Input(shape=(None, None, 3))  # 输入图像 高度和宽度可变 通道数为3
    roi_input = Input(shape=(None, 4))  # RPN生成的ROIs 每个ROI由4个坐标组成

    # 加载预训练的ResNet50模型 这里只获取特征提取部分
    base_layers = ResNet50(inputs, include_top=False, weights='imagenet', pooling=None)

    # 计算锚点框的总数
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)

    # 获取RPN模型部分
    rpn = get_rpn(base_layers.output, num_anchors)  # 使用base_layers的输出作为输入
    model_rpn = Model(inputs, rpn[:2])  # 只取RPN的前两个输出（类别预测和边界框预测）

    # 获取分类器模型部分 使用特征图和ROIs作为输入
    classifier = get_classifier(base_layers.output, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)  # 分类器模型需要同时接收图像和ROIs

    model_all = Model([inputs, roi_input], rpn[:2] + classifier)

    return model_rpn, model_classifier, model_all

def get_predict_model(config,num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None,None,1024))

    base_layers = ResNet50(inputs)
    # 获取RPN模型部分
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)
    # 获取分类器模型部分
    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn,model_classifier_only