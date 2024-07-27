#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam


# In[2]:


def Alextnet(input_shape=(224,224,3), output_shape = 2):
    model = Sequential()
    #开始定义网络结构
    '''
    1. Conv2D, 步长大小4x4，卷积核大小11 输出特征层位96层，所以输出的shape是 55，55,96 本次为简化模型采用48卷积核
    2. Batchnormalization 为快速收敛
    3. Maxpooling2D 步长为2，输出是27 27 96
    4. Conv2D, 步长大小1x1，卷积核大小5 输出特征层位256层 输出的shape是（27,27,256）本次简化模型输出为128特征层
    5. Batchnormalization 为快速收敛
    6. Maxpooling2D 步长为2，输出是13,13,256
    7. Conv2D
    8. Conv2D
    9. Conv2D
    10 Maxpooling2D
    '''
    model.add(Conv2D(filters=48, kernel_size=(11,11), strides=(4,4), padding='valid', input_shape = input_shape, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = 'valid'))
    
    model.add(Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding='same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = 'valid'))    
    
    model.add(Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'relu'))
    
    model.add(Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'relu'))
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = 'valid'))    
    
    '''
    最后为全连接层
    1. 展平
    2. 全连接层 激活函数relu 附带dropout
    3. 全连接层 激活函数relu附带dropout
    4. 全连接层 激活函数softmax 输出
    '''
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(output_shape, activation = 'softmax'))
    
    return model


# In[ ]:




