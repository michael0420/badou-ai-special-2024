#Resnet---深度残差网络
from tensorflow.python.keras.layers import Conv2D,Dense,MaxPooling2D,ZeroPadding2D,AveragePooling2D,Activation,BatchNormalization,Flatten
from  tensorflow.python.keras import layers
from  tensorflow.python.keras import Input
from  tensorflow.python.keras import Model
import cv2
import numpy as np
from keras.applications.resnet50 import preprocess_input
import keras.applications.imagenet_utils  as u


#定义identity_block
def identity_block(inputs,ks,filters,stage,block):
    filter1,filter2,filter3=filters  #卷积核通道数
    conv_name_base='res'+str(stage)+block+'_branch'
    bn_name_base='bn'+str(stage)+block+'_branch'

    x=Conv2D(filter1,(1,1),name=conv_name_base+'2a')(inputs)
    x=BatchNormalization(name=bn_name_base+'2a')(x)
    x=Activation('relu')(x)

    x=Conv2D(filter2,ks,padding='same',name=conv_name_base+'2b')(x)
    x=BatchNormalization(name=bn_name_base+'2b')(x)
    x=Activation('relu')(x)

    x=Conv2D(filter3,(1,1),name=conv_name_base+'2c')(x)
    x=BatchNormalization(name=bn_name_base+'2c')(x)

    x=layers.add([x,inputs])
    x=Activation('relu')(x)
    return x

#定义conv_block
def conv_block(inputs,ks,filters,stage,block,strides=(2,2)):
    filter1,filter2,filter3=filters  #卷积核通道数
    conv_name_base='res'+str(stage)+block+'_branch'
    bn_name_base='bn'+str(stage)+block+'_branch'

    x=Conv2D(filter1,(1,1),strides=strides,name=conv_name_base+'2a')(inputs)
    x=BatchNormalization(name=bn_name_base+'2a')(x)
    x=Activation('relu')(x)

    x=Conv2D(filter2,ks,padding='same',name=conv_name_base+'2b')(x)
    x=BatchNormalization(name=bn_name_base+'2b')(x)
    x=Activation('relu')(x)

    x=Conv2D(filter3,(1,1),name=conv_name_base+'2c')(x)
    x=BatchNormalization(name=bn_name_base+'2c')(x)

    shortcut=Conv2D(filter3,(1,1),strides=strides,name=conv_name_base+'l')(inputs)
    shortcut=BatchNormalization(name=bn_name_base+'l')(shortcut)

    x=layers.add([x,shortcut])
    x=Activation('relu')(x)
    return x

#定义模型结构
def  Resnet(input_shape=[224,224,3],num=1000):
    img_input=Input(shape=input_shape)   #输入图片格式为 224,224,3
    x=ZeroPadding2D((3,3))(img_input)   #外面三层填充0  此时shape为230,230,3

    x=Conv2D(64,(7,7),strides=(2,2),name='conv1')(x)  #此时shape为112,112,64
    x=BatchNormalization(name='conv1-bn')(x)
    x=Activation('relu')(x)
    x=MaxPooling2D((3,3),strides=(2,2))(x)   #此时shape为56,56,64

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')  #56,56,256

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')  #conv_block中有个步长为2的卷积操作  因此此时的shape为28,28,512
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')  #28,28,512

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f') #14,14,1024

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c') #7,7,2048

    x=AveragePooling2D((7,7),name='avg_pool')(x)   #1,1,2048
    x=Flatten()(x)
    x=Dense(num,activation='softmax',name='fc1000')(x)   #1,1000

    model=Model(img_input,x,name='resnet50')
    model.load_weights('./resnet50_weights_tf_dim_ordering_tf_kernels.h5')  #载入已有权重模型

    return model


if __name__=='__main__':
    model=Resnet()
    model.summary()
    img=cv2.imread('F:/PNG/bike.jpg')
    img=cv2.resize(img,(224,224))
    img=np.expand_dims(img,0)
#preprocess_input(),这是tensorflow下keras自带的类似于一个归一化的函数
    img=preprocess_input(img)
    print(img.shape)
    predict=model.predict(img)
    print('Predicted:', u.decode_predictions(predict))




















