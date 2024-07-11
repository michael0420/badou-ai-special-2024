# 2. resent50
from __future__ import print_function
import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

# stage:区分网络不同阶段
def identity_block(input_tensor,kernel_size,filters,stage,block):
    filter1,filter2,filter3 = filters

    # 卷积层 归一化层命名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 生成卷积层 归一化层
    x = Conv2D(filter1,(1,1),name=conv_name_base+'2a')(input(input_tensor))
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, name=conv_name_base + '2b')(input(x))
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(input(input_tensor))
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 卷积结果与捷径输出相加
    x = layers.add([x,input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block(input_tesor,kernel_size,filters,stage,block,strides=(2,2)):
    filter1,filter2,filter3 = filters

    # 卷积层 归一化层命名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 正常卷积的部分
    x = Conv2D(filter1,(1,1),strides=strides,name=conv_name_base+'2a')(input_tesor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, strides=strides, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), strides=strides, padding='same', name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filter3,(1,1),strides=strides,name=conv_name_base+'1')(input_tesor)
    shortcut = BatchNormalization(name=bn_name_base+'1')(shortcut)

    x = layers.add([x,shortcut])
    x = Activation('relu')(x)

    return x

# 定义网络结构 classes:类别输俩个
def resnet50(input_shape=[224,224,3],classes = 1000):
    img_input = Input(shape=input_shape) # 创建输入层
    x = ZeroPadding2D((3,3))(img_input) # 零填充

    # 定义网络结构

    x = Conv2D(64,(7,7),strides=(2,2),name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    x = conv_block(x,3,[64,64,256],stage=2,block='a',strides=(1,1))
    x = identity_block(x,3,[64,64,256],stage=2,block='b')
    x = identity_block(x,3,[64,64,256],stage=2,block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256,256,1024], stage=4, block='a')
    x = identity_block(x, 3,[256,256,1024], stage=4, block='b')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='c')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='d')
    x = identity_block(x, 3, [256,256,1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3,[512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 收尾
    x = AveragePooling2D((7,7),name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(classes,activation='softmax',name='fc1000')(x)

    model = Model(img_input,x,name='resent50')
    model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5') #加载预训练权重

    return model

if __name__ == '__main__':
    model = resnet50()

    model.summary() #打印模型的摘要信息:层的形状，输出形状，参数数量，查看整个模型的结构和参数分布
    img_path = "cat.jpg"
    img = image.load_img(img_path,target_size=(224,224)) # 加载图像，改变尺寸
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)

    print('Input shape:',x.shape)
    preds = model.predict(x)
    print("Predicted:",decode_predictions(preds)) #跑了半天没出结果 T^T
