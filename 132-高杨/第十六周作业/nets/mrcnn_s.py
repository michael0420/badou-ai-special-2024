from keras.layers import Input,Conv2D,MaxPooling2D,Activation,Reshape,Conv2DTranspose,BatchNormalization,UpSampling2D,Add,Lambda,Concatenate,Dense,TimeDistributed
from keras.models import Model
from Badou.第十六周8月3日.nets.resnet import get_resnet
from Badou.第十六周8月3日.nets.layers import ProposalLayer,PyramidROIAlign,DetectionLayer,DetectionTargetLayer
from Badou.第十六周8月3日.nets.mrcnn_training import *
from Badou.第十六周8月3日.utils.anchors import get_anchors
from Badou.第十六周8月3日.utils.utils import norm_boxes_graph,parse_image_meta_graph
import tensorflow as tf
import numpy as np

def rpn_graph(feature_map , anchors_per_location):

    # 五个不同大小的特征层传入到RPN中 ，获得建议框

    shared = Conv2D(512,(3,3),padding='same',activation='relu')(feature_map)
    x = Conv2D(2 * anchors_per_location,(1,1),padding='valid',activation='linear')(shared)

    # batchsize num_anchors 2 是这样的形状
    rpn_class_logits = Reshape([-1,2])(x)

    # 形状不变 但是值会在0-1之间
    rpn_probs = Activation('softmax')(rpn_class_logits)

    x = Conv2D(anchors_per_location * 4, (1,1),padding='valid',
               activation='linear'
               )(shared)

    rpn_bbox = Reshape([-1,4])(x)


    return [rpn_class_logits,rpn_probs,rpn_bbox]


def bulid_rpn_model(anchors_per_location,depth):
    input_feature_map = Input(shape=[None,None,depth],
                              name='input_rpn_feature_map'
                              )
    outputs = rpn_graph(input_feature_map,anchors_per_location)
    return Model([input_feature_map],outputs)


def fpn_classifier_grap(rois , feature_maps, image_meta,
                        pool_size,num_classes, train_bn = True,
                        fc_layers_size = 1024
                        ):

    # shape  batch num_rois pool_size pool_size channels
    x = PyramidROIAlign([pool_size,pool_size],name='roi_align_classifier')([rois,image_meta] + feature_maps)

    # batch num_rois 1 1 fc  相当于两次全连接但是共享参数
    x = TimeDistributed(Conv2D(fc_layers_size,(pool_size,pool_size),padding='valid'),name='mrcnn_class_conv1')(x)
    x = TimeDistributed(BatchNormalization() , name='mrcnn_class_bn1')(x,training = train_bn)
    x = Activation('relu')(x)

    # shape   batch  num 1  1 fc
    x = TimeDistributed(Conv2D(fc_layers_size,(1,1)),name='mrcnn_class_bn2')(x,training = train_bn)
    x = Activation('relu')(x)

    # shape batch num_rois fc
    shared = Lambda(lambda x:K.squeeze(K.squeeze(x , 3),2),
                    name='pool_squeeze'
                    )(x)

    # Classifier head
    mrcnn_class_logits = TimeDistributed(Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = TimeDistributed(Activation('softmax'),name='mrcnn_class_logits')(mrcnn_class_logits)

    # bbox head
    # 这个预测结果对先验框进行调整
    x = TimeDistributed(Dense(num_classes * 4 ,activation='linear'))(shared)
    mrcnn_bbox = Reshape((-1,num_classes,4),name='mrcnn_bbox')(x)

    return mrcnn_class_logits , mrcnn_probs,mrcnn_bbox


def build_fpn_mask_graph(rois , feature_maps,image_meta,
                         pool_size,num_classes,train_bn=True
                         ):
    # 利用建议框在特征层上进行截取
    # shape  batch num_rois mask_pool_size mask_pool_size channels
    x = PyramidROIAlign([pool_size,pool_size],)([rois,image_meta] + feature_maps)

    x = TimeDistributed(Conv2D(256,(3,3),padding='same'))(x)
    x = TimeDistributed(BatchNormalization(),)(x,training = train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(256,(3,3),padding='same'))(x)
    x = TimeDistributed(BatchNormalization(),)(x,training = train_bn)
    x = Activation('relu')(x)

    # shape  batch  num_rois  mask pool size  mask pool size  channels


    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'))(x)
    x = TimeDistributed(BatchNormalization(), )(x, training=train_bn)
    x = Activation('relu')(x)


    # shape  batch  num_rois  2*mask pool size  2*mask pool size  channels
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'))(x)
    x = TimeDistributed(BatchNormalization(), )(x, training=train_bn)
    x = Activation('relu')(x)


    # 上采样  对
    x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2,activation='relu'))(x)
    x = TimeDistributed(Conv2D(num_classes,(1,1),strides=1,activation='sigmoid'))(x)

    return x





def get_predict_model(config):
    h,w = config.IMAGE_SHAPE[:2]

    if h/ 2**6 != int(h / 2 **6) or w / 2**6 !=int(w / 2**6):
        raise Exception('Image size must dividable by 2 at least 6 times'\
                        'for example, use 256 320 384')

    input_image = Input(shape=[None,None,config.IMAGE_SHAPE[2]],name='input_image')

    # 元信息 包括图片id 原始图像形状的是三个元素 调整大小后的图像形状三个元素 ，window窗口坐标的四个元素，scale图像缩放因子
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE],name='input_meta_image')

    # 输入的先验框
    input_anchors= Input(shape=[None,4],name='input_anchors')

    # 获得Resnet中不同层
    _,C2,C3,C4,C5 = get_resnet(input_image,stage5=True,train_bn=config.TRAIN_BN)

    #组成特征金字塔结构  P5的长宽一共压缩了5次
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE,(1,1))(C5)

    P4 = Add(name='fpn_p4dd')([
        UpSampling2D(size=(2,2))(P5),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE,(1,1))(C4)
    ])

    P3 = Add(name='fpn_p3dd')(
        [
            UpSampling2D(size=(2,2))(P4),
            Conv2D(config.TOP_DOWN_PYRAMID_SIZE,(1,1))(C3)
        ]
    )

    P2 = Add(name='fpn_p2dd')(

        [
            UpSampling2D(size=(2,2))(P3),
            Conv2D(config.TOP_DOWN_PYRAMID_SIZE,(1,1))(C2)

        ]
    )


    # 各进行256通道卷积
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE,(3,3),padding='same')(P2)
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE,(3,3),padding='same')(P3)
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE,(3,3),padding='same')(P4)
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE,(3,3),padding='same')(P5)
    # 在建议框网络中又一个P6用于获取建议框
    # Height / 64 width / 64，256
    P6 = MaxPooling2D(pool_size=(1,1),strides=2,)(P5)

    # P2,P3,P4,P5,P6 可以用于生成建议框
    rpn_feature_maps = [P2,P3,P4,P5,P6]
    # P2-P5用于获取建议框
    mrcnn_feature_maps = [P2,P3,P4,P5]

    anchors = input_anchors
    #
    rpn = bulid_rpn_model(len(config.RPN_ANCHOR_RATIOS),config.TOP_DOWN_PYRAMID_SIZE)


    rpn_class_logits, rpn_class , rpn_bbox = [] , [] , []

    for p in  rpn_feature_maps:
        # 对5个特征层得到的结果进行堆叠
        logits,classes,bbox = rpn([p])
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)

    #
    rpn_class_logits = Concatenate(axis=1,name='rpn_class_logits')(rpn_class_logits)
    rpn_class = Concatenate(axis=1)(rpn_class)
    rpn_bbox = Concatenate(axis=1)(rpn_bbox)

    # 此时获得的rpn_class_logits  class rpn bbox 维度分别是\
    #  Batch_size, num_anchors, 2        Batch_size, num_anchors, 2        Batch_size, num_anchors, 4

    # 训练是 2000
    proposal_count = config.POST_NMS_ROIS_INFERENCE

    rpn_rois = ProposalLayer(proposal_count,nms_threshold=config.RPN_NMS_THRESHOLD,
                             name='roi',
                             config=config
                             )([rpn_class,rpn_bbox,anchors])

    # 获得classifier的结果
    mrcnn_class_logits , mrcnn_class , mrcnn_bbox =\
        fpn_classifier_grap(rpn_rois,mrcnn_feature_maps,
                            input_image_meta,config.POOL_SIZE,
                            config.NUM_CLASSES,
                            train_bn=config.TRAIN_BN,
                            fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE
                            )
    detections = DetectionLayer(config,)[rpn_rois,mrcnn_class,mrcnn_bbox,input_image_meta]
    detections_boxes = Lambda(lambda x:x[...,:4])(detections)

    # 获取mask 结果
    mrcnn_mask = build_fpn_mask_graph(detections_boxes,mrcnn_feature_maps,
                                      input_image_meta,
                                      config.MASK_POOL_SIZE,
                                      config.NUM_CLASSES,
                                      train_bn=config.TRAIN_BN
                                      )
    model = Model([input_image , input_image_meta,input_anchors],
                  [detections , mrcnn_class,mrcnn_bbox,
                   mrcnn_mask,rpn_rois,rpn_class,rpn_bbox
                   ]
                  )

    return model









