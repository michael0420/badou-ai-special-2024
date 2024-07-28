'''

MoblieNet 深度可分离卷积 ， 网络部分

'''
import numpy as np
from  keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D,Dense,Dropout,Conv2D,Input,MaxPooling2D,AveragePooling2D,Activation,BatchNormalization,GlobalAveragePooling2D,Reshape
from keras.applications.imagenet_utils import  decode_predictions
from keras import  backend as K


def relu6(x):
    return  K.relu(x,max_value=6)

def conv_block(inputs,filters,kernel=(3,3),strides=(1,1)):
    x = Conv2D(
        filters,kernel,
        padding='same',
        use_bias=False,
        strides=strides
    )(inputs)

    x = BatchNormalization()(x)
    return Activation(relu6)(x)


def Mobelnet(input_shape=[224,224,3],depth_multiplier=1,
              dropout = 1e-3,
              classes = 1000
              ):
    img_input = Input(shape=input_shape)

    # 224,224,3 ->112,112,32
    x = conv_block(img_input,32,strides=(2,2))

    x = depthwise_conv_block(x,64,depth_multiplier)

    x = depthwise_conv_block(x, 128, depth_multiplier,strides=(2,2))
    x = depthwise_conv_block(x, 128, depth_multiplier)

    x = depthwise_conv_block(x,256,depth_multiplier,strides=(2,2))
    x = depthwise_conv_block(x,256,depth_multiplier)

    x = depthwise_conv_block(x,512,depth_multiplier,strides=(2,2))

    # 14 14 512 14 14 512
    x = depthwise_conv_block(x,512,depth_multiplier)
    x = depthwise_conv_block(x,512,depth_multiplier)
    x = depthwise_conv_block(x,512,depth_multiplier)
    x = depthwise_conv_block(x,512,depth_multiplier)
    x = depthwise_conv_block(x,512,depth_multiplier)

    x = depthwise_conv_block(x,1024,depth_multiplier,strides=(2,2))

    x = depthwise_conv_block(x,1024,depth_multiplier)

    # 7,7,1024  -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,1024))(x)
    x = Dropout(dropout)(x)
    x = Conv2D(classes,(1,1),padding='same')(x)
    x = Activation('softmax')(x)
    x = Reshape((classes,))(x)

    inputs = img_input

    model = Model(inputs,x,name='MobelNet')
    model.load_weights('mobilenet_1_0_224_tf.h5')

    return model

def preprocess_input(x):
    x /=255.
    x -=0.5
    x *=2
    return x

def depthwise_conv_block(inputs,pointwise_conv_filters,
              depth_multiplier =1,
              strides=(1,1)
              ):

    x = DepthwiseConv2D((3,3),padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False
                        )(inputs)

    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    x = Conv2D(pointwise_conv_filters,(1,1),padding='same',
               use_bias=False,
               strides=(1,1)
               )(x)
    x = BatchNormalization()(x)
    return  Activation(relu6)(x)

if __name__ == '__main__':
    model = Mobelnet()

    img_path = 'plane.jpg'
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)

    print('input shape:',x.shape)

    preds = model.predict(x)


    print(np.argmax(preds))

    print('Predicted:',decode_predictions(preds,1))


