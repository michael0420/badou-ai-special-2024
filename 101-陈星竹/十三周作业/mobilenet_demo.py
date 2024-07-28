import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


# relu6
# 限制激活值的最大值
def relu6(x):
    return K.relu(x,max_value=6)
# 卷积块
def _conb_block(inputs,filters,kernel=(3,3),strides=(1,1)):
    x = Conv2D(filters,kernel,padding='same',use_bias=False,strides=strides,name = 'conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6,name='conv1_relu')(x)

# 深度卷积块
def _depthwise_conv_block(inputs,pointwise_conv_filters,depth_multiplier=1,strides=(1,1),block_id = 1):
    x = DepthwiseConv2D((3,3),padding='same',depth_multiplier=depth_multiplier,strides=strides,use_bias=False,name='conv_dw_'+str(block_id))(inputs)
    x = BatchNormalization(name='conv_dw_bn_'+str(block_id))(x)
    x = Activation(relu6,name='conv_dw_relu_'+str(block_id))(x)
    x = Conv2D(pointwise_conv_filters,(1,1),strides=(1,1),padding='same',use_bias=False,name='conv_pw_'+str(block_id))(x)
    x =BatchNormalization(name='conv_pw_bn_'+str(block_id))(x)

    return Activation(relu6,name="conv_pw_relu_"+str(block_id))(x)

def MobileNet(input_shape=[224,224,3],depth_multipler=1,dropout=1e-3,classes=1000):
    img_input = Input(shape=input_shape)

    # 244x244x3 -> 112x112x32
    x = _conb_block(img_input,32,strides=(2,2))
    # 112*112*32 -> 112*112*64
    x = _depthwise_conv_block(x,64,depth_multipler,block_id=1)
    # 112*112*64 -> 56*56*128
    x = _depthwise_conv_block(x,128,depth_multipler,strides=(2,2),block_id=2)
    # 56*56*128 -> 56*56*128
    x = _depthwise_conv_block(x,128,depth_multipler,block_id=3)
    # 56*56*128 ->  28*28*256
    x = _depthwise_conv_block(x,256,depth_multipler,strides=(2,2),block_id=4)
    # 28*28*128 -> 28*28*256
    x = _depthwise_conv_block(x,256,depth_multipler,block_id=5)
    # 28*28*256 -> 14*14*512
    x = _depthwise_conv_block(x,512,depth_multipler,strides=(2,2),block_id=6)
    # 14*14*512 ->14*14*512 x5
    x = _depthwise_conv_block(x,512,depth_multipler,block_id=7)
    x = _depthwise_conv_block(x,512,depth_multipler,block_id=8)
    x = _depthwise_conv_block(x,512,depth_multipler,block_id=9)
    x = _depthwise_conv_block(x,512,depth_multipler,block_id=10)
    x = _depthwise_conv_block(x,512,depth_multipler,block_id=11)
    # 14*14*512 -> 7*7*1024
    x = _depthwise_conv_block(x,1024,depth_multipler,strides=(2,2),block_id=12)
    x = _depthwise_conv_block(x,1024,depth_multipler,block_id=13)

    # 拍扁
    # 7*7*1024 -> 1*1*1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,1024),name='reshape_1')(x)
    x = Dropout(dropout,name='dropout1')(x)
    x = Conv2D(classes,(1,1),strides=(1,1),padding='same',name='conv_pred')(x)
    x = Activation('softmax',name='act_softmax')(x)
    x = Reshape((classes,),name='reshape_2')(x)
    model = Model(img_input, x, name='mobilenet_1_0_224_tf')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print("Raw predictions:", decode_predictions(preds))

