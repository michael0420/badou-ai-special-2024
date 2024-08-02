#Mobilenet
from tensorflow.python.keras.layers import Conv2D,Dense,Dropout,Reshape,GlobalMaxPooling2D,Activation,BatchNormalization,Flatten,DepthwiseConv2D
from  tensorflow.python.keras import Input
from  tensorflow.python.keras import Model
import cv2
import numpy as np
from keras import backend as K
import keras.applications.imagenet_utils  as u

#定义relu6函数
def relu6(x):
    return K.relu(x,max_value=6)

#定义卷积块---conv-bn--relu
def conv_blcok(inputs,filters,ks=(3,3),strides=(1,1)):
    x=Conv2D(filters,ks,padding='same',use_bias=False,strides=strides,name='conv1')(inputs)
    x=BatchNormalization(name='con1_bn')(x)
    x=Activation(relu6,name='conv1_relu')(x)
    return x

#定义depthwise+pointwise卷积块
def depthwise_conv_block(inputs,pointwise_conv_filters,depth_multiplier=1,strides=(1,1),block_id=1):
    x=DepthwiseConv2D((3,3),padding='same',depth_multiplier=depth_multiplier,strides=strides,use_bias=False,name='conv_dw_%d'%block_id)(inputs)
    x=BatchNormalization(name='conv_dw_%d_bn'%block_id)(x)
    x=Activation(relu6,name='conv_dw_%d_relu'%block_id)(x)

    x=Conv2D(pointwise_conv_filters,(1,1),padding='same',use_bias=False,strides=strides,name='conv_pw_%d'%block_id)(x)
    x=BatchNormalization(name='conv_pw_%d_bn'%block_id)(x)
    x=Activation(relu6,name='conv_pw_%d_relu'%block_id)(x)
    return x


def preprocess_inputs(x):
    x=x/255.   #(0,1)
    x=x-0.5    #(-0.5,0.5)
    x=x*2.     #(-1,1)
    return x

#定义mobilenet模型结构
def MobileNet(inputs_shape=[224,224,3],depth_multiplier=1,dropout=1e-3,classes=1000):
    img_input=Input(shape=inputs_shape)  #输入图片大小224,224,3
    x=conv_blcok(img_input,32,strides=(2,2))   #112,112,32

    x=depthwise_conv_block(x,64,depth_multiplier,block_id=1) #112,112,64
    x=depthwise_conv_block(x,128,depth_multiplier,strides=(2,2),block_id=2)  #56,56,128
    x=depthwise_conv_block(x,128,depth_multiplier,block_id=3)   ##56,56,128
    x=depthwise_conv_block(x,256,depth_multiplier,strides=(2,2),block_id=4)  #28,28,256
    x=depthwise_conv_block(x,256,depth_multiplier,block_id=5)  #28,28,256
    x=depthwise_conv_block(x,512,depth_multiplier,strides=(2,2),block_id=6)  #14,14,512

    x=depthwise_conv_block(x,512,depth_multiplier,block_id=7)
    x=depthwise_conv_block(x,512,depth_multiplier,block_id=8)
    x=depthwise_conv_block(x,512,depth_multiplier,block_id=9)
    x=depthwise_conv_block(x,512,depth_multiplier,block_id=10)
    x=depthwise_conv_block(x,512,depth_multiplier,block_id=11)  #14,14,512

    x=depthwise_conv_block(x,1024,depth_multiplier,strides=(2,2),block_id=12)  #7,7,1024
    x=depthwise_conv_block(x,1024,depth_multiplier,block_id=13)  #7,7,1024

    x=GlobalMaxPooling2D()(x)   #GlobalAveragePooling2D 全局平均池化
    x=Reshape((1,1,1024),name='reshape1')(x)  #1,1,1024
    x=Dropout(dropout,name='dropout')(x)
    x=Conv2D(classes,(1,1),padding='same',name='conv_predict')(x)
    x=Activation('softmax',name='softmax')(x)
    x=Reshape((classes,),name='reshape2')(x)

    inputs=img_input
    model=Model(inputs,x,name='mobilenet')
    model.load_weights('./mobilenet_1_0_224_tf.h5')  #导入训练好的权重

    return model


if __name__=='__main__':
    model=MobileNet()
    predict_img=cv2.imread('F:/PNG/bike.jpg')
    predict_img=cv2.resize(predict_img,(224,224))
    predict_img=np.expand_dims(predict_img,0)
    predict_img=preprocess_inputs(predict_img)
    print(predict_img.shape)
    predict=model.predict(predict_img)
    print(np.argmax(predict))
    print('Predicted:', u.decode_predictions(predict,1))













