from keras.layers import Input,Conv2D,Reshape,TimeDistributed,Flatten,Dense
from nets.resnet import ResNet50, classifier_layers
from keras.models import Model
from nets.RoiPooling import RoiPoolingConv

# RPN网络
def get_rpn(base_layers,num_anchors):
    # rpn网络的首个卷积：3*3，输入输出特征图像相同，使用正态分布进行权重初始化
    x = Conv2D(512,(3,3),padding='same',activation='relu',kernel_initializer='normal',name='rpn_conv1')(base_layers)

    # 分类预测：这里的通道为9，大于0.5为正样本，小于0.5为负样本。二分类的常用做法。使用sigmoid激活函数进行二分类，使用均匀分布来初始化权重
    x_class = Conv2D(num_anchors,(1,1),activation='sigmoid',kernel_initializer='uniform',name='rpn_out_class')(x)
    # 回归预测：线性激活，全零初始化权重
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    # 使用Reshape层将分类预测结果和回归预测结果调整为适当的形状。(-1, 1)，表示将一维张量转换为二维张量
    x_class = Reshape((-1,1),name='classification')(x_class)
    x_regr = Reshape((-1,4),name='regression')(x_regr)

    return [x_class,x_regr,base_layers]

# 分类器网络：包含了ROIPooling层
def get_classifier(base_layers,input_rois,num_rois,nb_classes=21,trainable=False):
    """
    :param base_layers:基础层，通常是一个卷积神经网络模型的输出
    :param input_rois:输入的感兴趣区域（ROI）坐标
    :param num_rois:ROI的数量
    :param nb_classes:类别数量，默认为21
    :param trainable:是否可训练，默认为False。决定在训练过程中是否会更新这些权重
    :return:每个样本的每个感兴趣区域的类别概率分布和边界框回归值
    """
    pooling_regions = 14  # 池化区域的大小
    input_shape=(num_rois,14,14,1024)
    # 进行ROIPooling，将各box的形状统一   调用接口，输入为resnet50的输出和建议框，输出为建议框的池化特征
    out_roi_pool = RoiPoolingConv(pooling_regions,num_rois)([base_layers,input_rois])
    # 进行最后的分类，输入经过roi层的统一大小数据
    out = classifier_layers(out_roi_pool,input_shape=input_shape,trainable=True)
    # 展平为1维特征向量 out的形状变为(-1,1)
    out = TimeDistributed(Flatten())(out)
    # 分类
    out_class = TimeDistributed(Dense(nb_classes,activation='softmax', kernel_initializer='zero'), name='Dense_class_{}'.format(nb_classes))(out)
    # 回归：类别-1是为为了减去背景类别，类别*4表示每个类别的x y w h
    out_regress = TimeDistributed(Dense(4 * (nb_classes - 1), activation="linear", kernel_initializer="zero"), name="Dense_regress_{}".format(nb_classes))(out)
    return [out_class,out_regress]

# 构建模型，输入为config和num_classes，输出为model_rpn,model_classifier,model_all
def get_model(config,num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))  # roi_input的形状为(None,4),即建议框的坐标,4为(xmin,ymin,xmax,ymax)
    base_layers = ResNet50(inputs)  # 调用ResNet50，输入为inputs，输出为base_layers

    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios) # 计算建议框的数量
    rpn = get_rpn(base_layers, num_anchors) # 调用get_rpn，输入为base_layers和num_anchors，输出为rpn
    model_rpn = Model(inputs, rpn[:2])   # 构建model_rpn，输入为inputs，输出为rpn[:2]

    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)

    model_all = Model([inputs, roi_input], rpn[:2] + classifier)
    return model_rpn, model_classifier, model_all


# 构建用于目标检测的模型
def get_predict_model(config, num_classes):
    # 图像输入层
    inputs = Input(shape=(None,None,3))
    # 感兴趣区域输入层
    roi_input = Input(shape=(None,4))
    # 特征图输入层,1024为特征提取网络的输出通道数
    feature_map_input = Input(shape=(None,None,1024))
    # 图像特征输入层
    base_layers = ResNet50(inputs)
    # 计算锚框数量
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    # 获取区域提议网络模型
    rpn = get_rpn(base_layers, num_anchors)
    # 将输入张量与RPN模型连接起来,这样我们就可以使用这个模型进行目标检测任务
    model_rpn = Model(inputs, rpn)

    # 构建一个只有分类的模型，接受特征图、ROI输入，并输出分类结果
    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input,roi_input],classifier)

    return model_rpn,model_classifier_only
