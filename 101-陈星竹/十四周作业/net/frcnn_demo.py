'''
模型结构
'''
from nets.resnet import ResNet50,classifier_layers
from keras.layers import Conv2D,Input,TimeDistributed,Flatten,Dense,Reshape
from keras.models import Model
from nets.RoiPoolingConv import RoiPoolingConv

def get_rpn(base_layers,num_anchors):
    x = Conv2D(512,(3,3),padding='same',activation='relu',kernel_initializer='normal',name='rpn_conv1')(base_layers)
    #分类，num_anchors:输出通道
    x_class = Conv2D(num_anchors,(1,1),activation='sigmoid',kernel_initializer='uniform',name='rpn_out_class')(x)
    #回归
    x_regr = Conv2D(num_anchors*4,(1,1),activation='linear',kernel_initializer='zero',name='rpn_out_reg')(x)
    x_class = Reshape((-1,1),name='classification')(x_class)
    x_regr = Reshape((-1,4),name='regression')(x_regr) #-1表示自动计算维度，使得总元素数量不变（为什么要reshape）

    return [x_class,x_regr,base_layers]

def get_classifier(base_layers,
                   input_rois, #rpn画出的候选框
                   num_rois, #候选框数量
                   nb_classes=21, trainable=False):
    inputs = Input(shape=(None, None, 3))  # 输入任意大小的三通道图像
    roi_input = Input(shape=(None, 4))  # 表示任意数量的候选区域，每个区域有4个坐标值（x,y,w,h）
    pooling_regions = 14 #输出特征图的大小
    input_shape = (num_rois,14,14,1024)  #设定ROI之后的候选框大小固定为14x14x1024
    #调整候选框的形状后，返回一张合并的大张量
    out_roi_pool = RoiPoolingConv(pooling_regions,num_rois)([base_layers,input_rois])
    # 进入全连接层之前的准备
    out = classifier_layers(out_roi_pool,input_shape=input_shape,trainable=True)
    '''
    展平操作
    输入：out 的形状为 (batch_size, num_rois, height, width, channels)，假设为 (1, 32, 14, 14, 2048)。
    作用：TimeDistributed(Flatten()) 将 Flatten 层应用到每个ROI上。
    输出：每个ROI的特征图被展平成一维向量，输出的形状为 (batch_size, num_rois, 14 * 14 * 2048)，即 (1, 32, 401408)。
    '''
    out = TimeDistributed(Flatten())(out)
    '''
    全连接层
    输入：展平后的特征图，形状为 (batch_size, num_rois, 401408)。
    作用：TimeDistributed(Dense(nb_classes, activation='softmax')) 将 Dense 层应用到每个展平后的特征图上，进行分类操作。
    输出：每个ROI的类别预测，形状为 (batch_size, num_rois, nb_classes)，即 (1, 32, 21)。
    '''
    out_class = TimeDistributed(Dense(nb_classes,activation='softmax',kernel_initializer='zero'),
                          name='dense_class_{}'.format(nb_classes))(out)

    '''
    输入：展平后的特征图，形状为 (batch_size, num_rois, 401408)。
    作用：TimeDistributed(Dense(4 * (nb_classes-1), activation='linear')) 将 Dense 层应用到每个展平后的特征图上，进行回归操作。
    输出：每个ROI的边界框回归参数，形状为 (batch_size, num_rois, 4 * (nb_classes-1))，即 (1, 32, 80)。
    '''
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    # roi层之后的回归层和分类层
    return [out_class,out_regr]

def get_model(config,num_classes):
    inputs = Input(shape=(None,None,3)) # 输入任意大小的三通道图像
    roi_input = Input(shape=(None,4)) #表示任意数量的候选区域，每个区域有4个坐标值（x,y,w,h）
    base_layers = ResNet50(inputs) #模型最开始的基础特征提取层
    # 根据config中锚点比例和尺度计算锚点个数
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    #构建rpn网络
    rpn = get_rpn(base_layers,num_anchors)
    model_rpn = Model(inputs,rpn[:2]) # 输出为rpn网络的前两个输出（分类和回归）x_class,x_reg
    # 使用分类器网络对每个候选区域进行分类，并回归预测其精确位置。
    classifier = get_classifier(base_layers,roi_input,config.num_rois,nb_classes=num_classes,trainable=True)
    model_classifier = Model([inputs,roi_input],classifier) # 创建分类器模型，输入为图像层和roi输入层

    # 将所有模型组件整合在一起
    model_all = Model([inputs, roi_input], rpn[:2] + classifier)

    return model_rpn,model_classifier,model_all

def get_predict_model(config,num_classes):
    inputs = Input(shape=[None,None,3])
    roi_input = Input(shape=(None,4))
    feature_map_input = Input(shape=(None,None,1024)) # 特征图输入层结构
    base_layers = ResNet50(inputs)
    # 锚点尺度大小的数量（[128, 256, 512]） * 锚点长宽比的数量（[[1, 1], [1, 2], [2, 1]]） = 锚点数量
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers,num_anchors)
    model_rpn = Model(inputs,rpn[:2])
    classifier = get_classifier(feature_map_input,roi_input,config.num_rois,nb_classes=num_classes,trainable=True)
    model_classifier_only = Model([feature_map_input,roi_input],classifier)
    return model_rpn,model_classifier_only



