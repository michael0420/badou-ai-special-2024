from  tensorflow.python.keras.datasets import cifar10
import numpy as  np
from tensorflow.python.keras.utils import to_categorical
from  tensorflow.python.keras import layers
from  tensorflow.python.keras import models,optimizers
import matplotlib.pyplot as  plt
import cv2




#alexnet网络模型
def Alexnet():
    inputs=layers.Input(shape=(224,224,3))  #输入数据shape为（224,224,3）

    L1=layers.ZeroPadding2D((2,2))(inputs)   #输入数据周围填充0使shape变为（227,227,3）
    L1=layers.Conv2D(96,(11,11),strides=(4,4),padding='valid',activation='relu')(L1)  #卷积核（11,11,96），步长4，卷积后大小为（55,55,96）
    L1=layers.MaxPooling2D((3,3),strides=(2,2))(L1)  #池化后大小为（27,27,96）

    L2=layers.Conv2D(256,(5,5),activation='relu',padding='same')(L1)   #卷积后的大小为（27,27,256）
    L2=layers.MaxPooling2D((3,3),strides=(2,2))(L2)   #池化后的大小为（13,13,256）

    L3=layers.Conv2D(384,(3,3),activation='relu',padding='same')(L2)  #卷积后大小为（13,13,384）

    L4=layers.Conv2D(384,(3,3),activation='relu',padding='same')(L3)  #卷积后大小为（13,13,384）

    L5=layers.Conv2D(256,(3,3),activation='relu',padding='same')(L4)  #卷积后大小为（13,13,256）
    L5=layers.MaxPooling2D((3,3),strides=(2,2))(L5)   #池化后的大小为（6,6,256）

    fc=layers.Flatten()(L5)  #flatten后的shape为（6*6*256,1）

    fc1=layers.Dense(4096)(fc)  #全连接层  输出结点数4096 过这一层后的大小为（4096,1）
    fc1=layers.BatchNormalization()(fc1)   #layers.BatchNormalization,用于实现批量标准化,以提高模型的训练速度和稳定性，同时减少过拟合的风险
    fc1=layers.Activation('relu')(fc1)
    fc1=layers.Dropout(0.5)(fc1)

    fc2=layers.Dense(4096)(fc1)
    fc2=layers.BatchNormalization()(fc2)
    fc2=layers.Activation('relu')(fc2)
    fc2=layers.Dropout(0.5)(fc2)

    pridict=layers.Dense(10)(fc2)
    pridict=layers.BatchNormalization()(pridict)
    pridict=layers.Activation('softmax')(pridict)

    model=models.Model(inputs,pridict)
    omz=optimizers.Adam(lr=0.01)  #优化器选择Adam
    #model.compile()函数被设计为一个编译器，用于将模型的图形结构定义与计算引擎进行链接，以实现优化、损失函数的选择和训练过程的配置
    model.compile(optimizer=omz,loss='categorical_crossentropy',metrics=['acc'])
    model.summary()  #model.summary()输出模型各层的参数状况
    return model

#导入cifar10数据
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
print(x_train.shape)
print(y_train.shape)

#to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
Y_train=to_categorical(y_train)
Y_test=to_categorical(y_test)
print(y_train.shape,y_test[0])

#归一化
x_train_mean=np.mean(x_train)
x_test_mean=np.mean(x_test)

plt.imshow(x_train[0])
tt=np.zeros((224,224,3))
tt=cv2.resize(x_train[0],(224,224),interpolation=cv2.INTER_NEAREST)
plt.imshow(tt)
#plt.show()

x_train1=x_train[15000:20000]
x_test1=x_test[:500]
print(x_train1.shape,x_test1.shape)

#cifar10数据集图像尺寸为32，而alexnet的输入是224，所以要把样本都reshape成224
X_train1=np.zeros((x_train1.shape[0],224,224,x_train1.shape[3]))
X_test1=np.zeros((x_test1.shape[0],224,224,x_test1.shape[3]))
for i in range(x_train1.shape[0]):
    X_train1[i]=cv2.resize(x_train[i],(224,224),interpolation=cv2.INTER_NEAREST)
for i in range(x_test1.shape[0]):
    X_test1[i]=cv2.resize(x_test[i],(224,224),interpolation=cv2.INTER_NEAREST)
print(X_test1.shape,X_train1.shape)

'''
#训练+保存模型
model=Alexnet()
model.fit(X_train1,Y_train[10000:15000],batch_size=50,epochs=10)
model.save('alexnet_model.h5')
'''

#导入已训练好的模型
model=models.load_model('alexnet_model.h5')
'''
#原有模型继续训练
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train1,Y_train[15000:20000],batch_size=100,epochs=10)



#测试
test_loss,test_acc=model.evaluate(X_test1,Y_test[:500],verbose=1)
print('test_loss',test_loss)
print('test_acc',test_acc)
'''
#拿测试集的某一张图来推理
label_list={1:'airplane',2:'automobile',3:'bird',4:'cat',5:'deer',6:'dog',7:'frog',8:'horse',9:'ship',10:'truck'}
x_predict=X_test1[11:12]
print(x_predict.shape)
res=model.predict(x_predict)
res=np.argmax(res)
print('res',res)
img=x_test1[10][1:]
print(img.shape)
plt.imshow(img)
plt.show()
#top=y_test[res]
print('the thing of model think for the picture is:',label_list.get(res))




