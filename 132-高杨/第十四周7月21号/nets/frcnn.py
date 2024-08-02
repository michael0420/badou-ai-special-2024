from Badou.第十四周7月21号.nets.resnet import ResNet50,classifier_layers
from keras.layers import Conv2D,Input,TimeDistributed,Flatten,Dense,Reshape
from keras.models import Model
from keras.utils import np_utils
from Badou.第十四周7月21号.nets.RoiPoolingConv1 import RoiPoolingConv


def get_rpn(base_layers,num_anchors):

    x = Conv2D(512,(3,3),activation='relu',padding='same',kernel_initializer='normal')(base_layers)

    # 分两路走
    x_class = Conv2D(num_anchors,(1,1),activation='sigmoid',kernel_initializer='uniform')(x)
    x_reg = Conv2D(num_anchors*4,(1,1),activation='linear',kernel_initializer='zero')(x)

    x_class = Reshape((-1,1),name='classification')(x_class)
    x_reg = Reshape((-1,4),name='regression')(x_reg)

    return [x_class,x_reg,base_layers]


def get_classifier(base_layres,input_rois,num_rois,nb_classes=21,trainable=False):
    pooling_regions = 14
    input_shape = (num_rois,14,14,1024)
    out_roi_pool = RoiPoolingConv(pooling_regions,num_rois)([base_layres,input_rois])
    out = classifier_layers(out_roi_pool,input_shape=input_shape,trainable=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes,activation='softmax',kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4*(nb_classes-1),activation='linear',kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return  [out_class,out_regr]

def get_model(config,num_classes):
    inputs = Input(shape=(None,None,3))
    roi_input = Input(shape=(None,4))
    base_layers = ResNet50(inputs)

    # 9 3种比例的框   每个比例三个大小
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers,num_anchors)
    model_rpn = Model(inputs,rpn[:2])

    classifier = get_classifier(base_layers,roi_input,config.num_rois)
    model_classifier = Model([inputs,roi_input],classifier)

    model_all = Model([inputs,roi_input],rpn[:2]+classifier)

    return  model_rpn,model_classifier,model_all

def get_predict_model(config,num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None,None,1024))

    base_layers = ResNet50(inputs)
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)

    classifier = get_classifier(base_layers, roi_input, config.num_rois,nb_classes=num_classes,trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return  model_rpn,model_classifier_only




