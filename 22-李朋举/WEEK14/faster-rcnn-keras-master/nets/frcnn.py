from nets.resnet import ResNet50, classifier_layers
from keras.layers import Conv2D, Input, TimeDistributed, Flatten, Dense, Reshape
from keras.models import Model
from nets.RoiPoolingConv import RoiPoolingConv


def get_rpn(base_layers, num_anchors):
    # 在基础网络的输出上添加一个卷积层，卷积核大小为 3x3，输出通道数为 512，激活函数为 relu，卷积核初始化方式为 normal，并将该层命名为 rpn_conv1
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)  # Tensor("rpn_conv1/Relu:0", shape=(?, ?, ?, 512), dtype=float32)

    # 分支1 卷积
    # 在卷积层 x 的输出上添加一个卷积层，卷积核大小为 1x1，输出通道数为锚框的数量，激活函数为 sigmoid，卷积核初始化方式为 uniform，并将该层命名为 rpn_out_class。该层用于预测每个锚框的类别。
    # x_class -> Tensor("rpn_out_class/Sigmoid:0", shape=(?, ?, ?, 9), dtype=float32)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    # 分支2 bboxregr
    # 在卷积层 x 的输出上添加一个卷积层，卷积核大小为 1x1，输出通道数为锚框数量的 4 倍，激活函数为 linear，卷积核初始化方式为 zero，并将该层命名为 rpn_out_regress。该层用于预测每个锚框的回归值。
    # x_regr -> Tensor("rpn_out_regress/BiasAdd:0", shape=(?, ?, ?, 36), dtype=float32)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    # 将卷积层 x_class 的输出进行重整形，使其变为 (-1, 1) 的形状，并将该层命名为 classification。
    # x_class -> Tensor("classification/Reshape:0", shape=(?, ?, 1), dtype=float32)
    x_class = Reshape((-1, 1), name="classification")(x_class)
    # 将卷积层 x_regr 的输出进行重整形，使其变为 (-1, 4) 的形状，并将该层命名为 regression。
    # x_regr -> Tensor("regression/Reshape:0", shape=(?, ?, 4), dtype=float32)
    x_regr = Reshape((-1, 4), name="regression")(x_regr)
    # 返回一个列表，其中包含了预测类别、预测回归值和基础网络的输出。
    return [x_class, x_regr, base_layers]


def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    # 定义了池化区域的大小
    pooling_regions = 14
    # 定义了输入的形状，包括感兴趣区域的数量、池化后的高度、宽度和深度。
    input_shape = (num_rois, 14, 14, 1024)  # {turple:4} (32, 14, 14, 1024)
    # 使用 RoiPoolingConv 层对基础层和感兴趣区域进行池化操作，得到池化后的特征图
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    # 对池化后的特征图进行进一步的处理
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)  # Tensor("time_distributed_1/Reshape_2:0", shape=(1, 32, 2048), dtype=float32)
    # 使用 TimeDistributed 和 Flatten 层将处理后的特征图展平。
    out = TimeDistributed(Flatten())(out)  # Tensor("time_distributed_1/Reshape_2:0", shape=(1, 32, 2048), dtype=float32)
    # 使用 TimeDistributed 和 Dense 层对展平后的特征图进行分类，输出每个类别的概率。
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)  # Tensor("dense_class_21/Reshape_1:0", shape=(1, 32, 21), dtype=float32)
    # 使用 TimeDistributed 和 Dense 层对展平后的特征图进行回归，输出每个类别的回归值。
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)  # Tensor("dense_regress_21/Reshape_1:0", shape=(1, 32, 80), dtype=float32)
    # 返回分类和回归的结果
    return [out_class, out_regr]

# 训练train中调用
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

# 载入模型，如果原来的模型里已经包括了模型结构则直接载入。否则先构建模型再载入
def get_predict_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))  # Tensor("input_1:0", shape=(?, ?, ?, 3), dtype=float32)
    roi_input = Input(shape=(None, 4))  # Tensor("input_2:0", shape=(?, ?, 4), dtype=float32)
    feature_map_input = Input(shape=(None, None, 1024))  # Tensor("input_3:0", shape=(?, ?, ?, 1024), dtype=float32)
    # [1] Conv layers 使用 ResNet50 作为基础网络，对输入的图像进行特征提取
    base_layers = ResNet50(inputs)  # Tensor("activation_40/Relu:0", shape=(?, ?, ?, 1024), dtype=float32)
    # 计算锚框的数量，根据配置文件中的锚框尺度和宽高比计算  num_anchors=3X3=9
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)  # 9
    # [2] RPN 区域生成网络  调用 get_rpn 函数获取区域提议网络（RPN）
    rpn = get_rpn(base_layers, num_anchors)
    # 构建 RPN 模型，将输入图像和锚框信息作为输入，输出预测的区域提议。
    model_rpn = Model(inputs, rpn)  # <keras.engine.training.Model object at 0x0000023CA915A9B0>
    # [3] ROI pooling 调用 get_classifier 函数获取分类器
    # [4] Classification
    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    # 构建分类器模型，将特征图和 ROI 信息作为输入，输出分类结果
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    # 返回构建好的 RPN 模型和分类器模型
    return model_rpn, model_classifier_only
