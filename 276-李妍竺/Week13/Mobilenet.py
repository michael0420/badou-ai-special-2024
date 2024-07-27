import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

def relu6(x):
    # 上限为6的relu
    return K.relu(x,max_value=6)

def conv_block(inputs,filters,kernel=(3,3),strides=(1,1)):
    '''
    Conv+BN+relu6
    :param inputs: 输入tensor
    :param filters: 卷积核数
    :param kernel: 卷积核大小
    :param strides: 步长
    :return: 模型结果
    '''

    x = Conv2D(filters,kernel,padding='same',use_bias=False,strides=strides,name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6,name='conv1_relu')(x)
    return x

def depthwise_conv_block(inputs,pointwise_conv_filters,depth_multiplier=1,strides=(1,1),block_id=1):
    '''
    深度可分离卷积
    :param inputs: 输入tensor
    :param pointwise_conv_filters:卷积核数
    :param depth_multiplier: 深度
    :param strides: 步长
    :param block_id: 块id
    :return: 模型
    '''
    x = DepthwiseConv2D((3,3),padding='same',depth_multiplier=depth_multiplier,strides=strides,use_bias=False,
                        name='conv_dw_%d'%block_id)(inputs)
    x = BatchNormalization(name='conv_dw_bn_%d'%block_id)(x)
    x = Activation(relu6,name='conv_dw_relu_%d'%block_id)(x)
    x = Conv2D(pointwise_conv_filters,(1,1),padding='same',use_bias=False,strides=(1,1),name='conv_pw_%d'%block_id)(x)
    x = BatchNormalization(name='conv_pw_bn_%d'%block_id)(x)
    x = Activation(relu6,name='conv_pw_relu_%d'%block_id)(x)
    return x

def MobileNet(input_shape=[224,224,3],depth_multiplier=1,dropout=1e-3,classes=1000):
    '''
    嵌入式深度可分离卷积
    :param input_shape:输入tensor尺度
    :param depth_multiplier: 深度数量
    :param dropout: dropout比例
    :param classes: 分类数
    :return: 模型
    '''

    img_input= Input(shape=input_shape)

    # 224,224,3 -> 112,112,32
    x = conv_block(img_input,32,strides=(2,2))
    # 112,112,32 -> 112,112,64
    x = depthwise_conv_block(x, 64, depth_multiplier, block_id=1)
    # 112,112,64 -> 56,56,128
    x = depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128
    x = depthwise_conv_block(x, 128, depth_multiplier, block_id=3)
    # 56,56,128 -> 28,28,256
    x = depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    # 28,28,256 -> 28,28,256
    x = depthwise_conv_block(x, 256, depth_multiplier, block_id=5)
    # 28,28,256 -> 14,14,512
    x = depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)

    # 14,14,512 -> 14,14,512
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Reshape((1,1,1024),name='reshape_1')(x)
    x = Dropout(dropout,name='dropout')(x)
    x = Conv2D(classes,(1,1),padding='same',name='conv_preds')(x)
    x = Activation('softmax',name='act_softmax')(x)
    x = Reshape((classes,),name='reshape_2')(x)   #(classes,) 表示一个一维向量

    # 准备输出模型
    inputs = img_input
    model = Model(inputs, x, name='mobilenet')

    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = MobileNet()
    model.summary()

    model.load_weights("mobilenet_1_0_224_tf.h5")

    img_path = 'elephant.jpg'
    img = image.load_img(img_path,target_size=(224,224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds))





