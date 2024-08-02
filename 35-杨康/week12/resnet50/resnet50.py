from __future__ import print_function
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPool2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Flatten, Activation, BatchNormalization
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input,decode_predictions

def identity_block(input_tensor,kernel_size,filters,stage,block):
    filters1,filters2,filters3 = filters
    conv_name_base = 'res' + str(stage) + block + 'branch_'
    bn_name_base = 'bn' + str(stage) + block + 'branch_'

    x = Conv2D(filters1, (1,1), name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x,input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + 'branch_'
    bn_name_base = 'bn' + str(stage) + block + 'branch_'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x,shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(input_shape=[224,224,3],class_num=1000):
    input = Input(input_shape)
    x = ZeroPadding2D((3,3))(input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3,3),strides=(2,2))(x)

    x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides=(1,1))
    x = identity_block(x,3,[64,64,256],stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(class_num, activation='softmax', name='fc1000')(x)

    model = Model(input, x, name='ResNet50')
    return model

if __name__ == "__main__":
    model = ResNet50()
    model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    model.summary()

    img = image.load_img('bike.jpg', target_size=(224,224))   #target_size需要和input_shape一样
    x = image.img_to_array(img)  #把图片数据转换成numpy格式
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pred = model.predict(x)

    print('prediction:', decode_predictions(pred))










