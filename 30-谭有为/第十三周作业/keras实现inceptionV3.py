from tensorflow.python.keras.layers import Activation,Dense,Input,Conv2D,MaxPooling2D,AveragePooling2D,BatchNormalization
from tensorflow.python.keras import layers
from  tensorflow.python.keras import Model
from keras.applications.inception_v3 import preprocess_input
import cv2
import numpy as np
import keras.applications.imagenet_utils  as u

#CONV+BN+AVTIVE 合为一个函数
def Conv2d_bn(x,filters,num_raw,num_col,padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name=name+'_bn'
        conv_name=name+'_conv'
    else:
        bn_name=None
        conv_name=None

    x=Conv2D(filters,(num_raw,num_col),strides=strides,padding=padding,use_bias=False,name=conv_name)(x)
    x=BatchNormalization(scale=False,name=bn_name)(x)
    x=Activation('relu',name=name)(x)
    return x

#构建模型结构
def InceptionV3(input_shape=[299,299,3],classes=1000):
    img_input=Input(shape=input_shape)   #输入shape 229,229,3

    x=Conv2d_bn(img_input,32,3,3,strides=(2,2),padding='valid')   #149,149,32
    x=Conv2d_bn(x,32,3,3,padding='valid')    #147,147,32
    x=Conv2d_bn(x,64,3,3)           #147,147,64
    x=MaxPooling2D((3,3),strides=(2,2))(x)  #73,73,64

    x=Conv2d_bn(x,80,1,1,padding='valid')  #73,73,80
    x=Conv2d_bn(x,192,3,3,padding='valid')   #71,71,192
    x=MaxPooling2D((3,3),strides=(2,2))(x)  #35,35,192

    # block1 part1  35,35,192---->35,35,256      block1 主要使用了不同大小的卷积核对输入进行卷积的方法
    branch1x1=Conv2d_bn(x,64,1,1)      #35,35,64

    branch5x5=Conv2d_bn(x,48,1,1)      #35,35,48
    branch5x5=Conv2d_bn(branch5x5,64,5,5)   #35,35,64

    branch3x3dbl=Conv2d_bn(x,64,1,1)  #35,35,64
    branch3x3dbl=Conv2d_bn(branch3x3dbl,96,3,3)   #35,35,96
    branch3x3dbl=Conv2d_bn(branch3x3dbl,96,3,3)   #35,35,96

    branch_pool=AveragePooling2D((3,3),strides=(1,1),padding='same')(x)  #35,35,192
    branch_pool=Conv2d_bn(branch_pool,32,1,1)     #35,35,32

    x=layers.concatenate([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3,name='mixed0')   #35,35,256

    #block1 part2  35,35,256-->35,35,288
    branch1x1=Conv2d_bn(x,64,1,1)    #35,35,64

    branch5x5=Conv2d_bn(x,48,1,1)    #35,35,48
    branch5x5=Conv2d_bn(branch5x5,64,5,5)   #35,35,64

    branch3x3dbl=Conv2d_bn(x,64,1,1)    #35,35,64
    branch3x3dbl=Conv2d_bn(branch3x3dbl,96,3,3)   #35,35,96
    branch3x3dbl=Conv2d_bn(branch3x3dbl,96,3,3)    #35,35,96

    branch_pool=AveragePooling2D((3,3),strides=(1,1),padding='same')(x)   #35,35,256
    branch_pool=Conv2d_bn(branch_pool,64,1,1)   #35,35,64

    x=layers.concatenate([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3,name='mixed1')  #35,35,288

    #block1 part3  35,35,288-->35,35,288
    branch1x1=Conv2d_bn(x,64,1,1)   #35,35,64

    branch5x5=Conv2d_bn(x,48,1,1)  #35,35,48
    branch5x5=Conv2d_bn(branch5x5,64,5,5)    #35,35,64

    branch3x3dbl=Conv2d_bn(x,64,1,1)   #35,35,64
    branch3x3dbl=Conv2d_bn(branch3x3dbl,96,3,3)  #35,35,96
    branch3x3dbl=Conv2d_bn(branch3x3dbl,96,3,3)  #35,35,96

    branch_pool=AveragePooling2D((3,3),strides=(1,1),padding='same')(x)   #35,35,288
    branch_pool=Conv2d_bn(branch_pool,64,1,1)      #35,35,64

    x=layers.concatenate([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3,name='mixed2')   #35,35,288

    #block 2    part1 35,35,288-->17,17,768    block 2 主要使用了利用 1,7和 7,1 代替7,7的卷积的方法
    branch3x3=Conv2d_bn(x,384,3,3,strides=(2,2),padding='valid')    #17,17,384

    branch3x3dbl=Conv2d_bn(x,64,1,1)   #35,35,64
    branch3x3dbl=Conv2d_bn(branch3x3dbl,96,3,3)   #35,35,96
    branch3x3dbl=Conv2d_bn(branch3x3dbl,96,3,3,strides=(2,2),padding='valid')  #17,17,96

    branch_pool=MaxPooling2D((3,3),strides=(2,2))(x)   #17,17,288

    x=layers.concatenate([branch3x3,branch3x3dbl,branch_pool],axis=3,name='mixed3')  #17,17,768

    #block2   part2  17,17,768-->17,17,768
    branch1x1=Conv2d_bn(x,192,1,1)  #17,17,192

    branch7x7=Conv2d_bn(x,128,1,1)   #17,17,128
    branch7x7=Conv2d_bn(branch7x7,128,1,7)   #17,17,128
    branch7x7=Conv2d_bn(branch7x7,192,7,1)   #17,17,192

    branch7x7dbl=Conv2d_bn(x,128,1,1)  #17,17,128
    branch7x7dbl=Conv2d_bn(branch7x7dbl,128,7,1)
    branch7x7dbl=Conv2d_bn(branch7x7dbl,128,1,7)
    branch7x7dbl=Conv2d_bn(branch7x7dbl,128,7,1)
    branch7x7dbl=Conv2d_bn(branch7x7dbl,192,1,7) #17,17,192

    branch_pool=AveragePooling2D((3,3),strides=(1,1),padding='same')(x)  #17,17,768
    branch_pool=Conv2d_bn(branch_pool,192,1,1)  #17,17,192

    x=layers.concatenate([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3,name='mixed4')  #17,17,768

    #block2  part3+part4  17,17,768-->17,17,768-->17,17,768
    for i in range(2):
        branch1x1=Conv2d_bn(x,192,1,1)

        branch7x7=Conv2d_bn(x,160,1,1)
        branch7x7=Conv2d_bn(branch7x7,160,1,7)
        branch7x7=Conv2d_bn(branch7x7,192,7,1)

        branch7x7dbl=Conv2d_bn(x,160,1,1)
        branch7x7dbl=Conv2d_bn(branch7x7dbl,160,7,1)
        branch7x7dbl=Conv2d_bn(branch7x7dbl,160,1,7)
        branch7x7dbl=Conv2d_bn(branch7x7dbl,160,7,1)
        branch7x7dbl=Conv2d_bn(branch7x7dbl,192,1,7)

        branch_pool=AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
        branch_pool=Conv2d_bn(branch_pool,192,1,1)

        x=layers.concatenate([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3,name='mixed'+str(5+i))

    #block2  part5  17,17,768-->17,17,768
    branch1x1=Conv2d_bn(x,192,1,1)

    branch7x7=Conv2d_bn(x,192,1,1)
    branch7x7=Conv2d_bn(branch7x7,192,1,7)
    branch7x7=Conv2d_bn(branch7x7,192,7,1)

    branch7x7dbl=Conv2d_bn(x,192,1,1)
    branch7x7dbl=Conv2d_bn(branch7x7dbl,192,7,1)
    branch7x7dbl=Conv2d_bn(branch7x7dbl,192,1,7)
    branch7x7dbl=Conv2d_bn(branch7x7dbl,192,7,1)
    branch7x7dbl=Conv2d_bn(branch7x7dbl,192,1,7)

    branch_pool=AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool=Conv2d_bn(branch_pool,192,1,1)

    x=layers.concatenate([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3,name='mixed7')

    #block3  part1  17,17,768--->8,8,1280    block 2 主要使用了利用 1,3和 3,1 代替3,3的卷积的方法
    branch3x3=Conv2d_bn(x,192,1,1)  #17,17,192
    branch3x3=Conv2d_bn(branch3x3,320,3,3,strides=(2,2),padding='valid')  #8,8,320

    branch7x7x3=Conv2d_bn(x,192,1,1)
    branch7x7x3=Conv2d_bn(branch7x7x3,192,1,7)
    branch7x7x3=Conv2d_bn(branch7x7x3,192,7,1)
    branch7x7x3=Conv2d_bn(branch7x7x3,192,3,3,strides=(2,2),padding='valid')   #8,8,192

    branch_pool=MaxPooling2D((3,3),strides=(2,2))(x)  #8,8,768

    x=layers.concatenate([branch3x3,branch7x7x3,branch_pool],axis=3,name='mixed8')    #8,8,1280

    #block3  part2 part3  8,8,1280--->8,8,2048-->8,8,2048
    for i in range(2):
         branch1x1=Conv2d_bn(x,320,1,1)   #8,8,,320

         branch3x3=Conv2d_bn(x,384,1,1) #8,8,384
         branch3x3x1=Conv2d_bn(branch3x3,384,1,3)  #8,8,384
         branch3x3x2=Conv2d_bn(branch3x3,384,3,1)  #8,8,384
         branch3x3=layers.concatenate([branch3x3x1,branch3x3x2],axis=3,name='mixed9_'+str(i)) #8,8,768

         branch3x3dbl=Conv2d_bn(x,448,1,1)   #8,8,448
         branch3x3dbl=Conv2d_bn(branch3x3dbl,384,3,3)  #8,8,384
         branch3x3dblx1=Conv2d_bn(branch3x3dbl,384,1,3)   #8,8,384
         branch3x3dblx2=Conv2d_bn(branch3x3dbl,384,3,1)  #8,8,384
         branch3x3dbl=layers.concatenate([branch3x3dblx1,branch3x3dblx2],axis=3,name='mixed9_'+str(i+2))  #8,8,768

         branch_pool=AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
         branch_pool=Conv2d_bn(branch_pool,192,1,1)   #8,8,192

         x=layers.concatenate([branch1x1,branch3x3,branch3x3dbl,branch_pool],axis=3,name='mixed'+str(9+i))  #8,8,2048

#平均池化后全连接
    x=layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x=Dense(classes,activation='softmax',name='pridictions')(x)

    inputs=img_input
    model=Model(inputs,x,name='inceptionV3')
    return model


if __name__=='__main__':
    model=InceptionV3()
    model.summary()
    model.load_weights('./inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
    #读入预测图片以及图片处理
    predict_img=cv2.imread('F:/PNG/bike.jpg')
    predict_img=cv2.resize(predict_img,(299,299))
    predict_img=np.expand_dims(predict_img,0)
    predict_img=preprocess_input(predict_img)
    print(predict_img.shape)
    predict=model.predict(predict_img)
    print('Predicted:', u.decode_predictions(predict))






