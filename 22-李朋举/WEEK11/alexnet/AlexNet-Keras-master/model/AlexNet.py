from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam

def AlexNet(input_shape=(224,224,3),output_shape=2):
    # AlexNet
    '''
    `model = Sequential()` 是 Keras 中用于创建顺序模型的代码。
    在 Keras 中，`Sequential` 模型是一种线性的、层叠的模型结构，其中的层按照顺序依次堆叠。通过将层添加到 `Sequential` 模型中，可以构建深度神经网络。
    以下是使用 `Sequential` 模型的一般步骤：
        1. 创建 `Sequential` 模型对象：使用 `Sequential()` 函数创建一个空的 `Sequential` 模型对象。
        2. 添加层：使用 `model.add()` 方法向模型中添加层。可以根据需要添加各种类型的层，如卷积层、全连接层、池化层等。
        3. 编译模型：使用 `model.compile()` 方法编译模型。在编译模型时，需要指定优化器、损失函数和评估指标等。
        4. 训练模型：使用 `model.fit()` 方法训练模型。在训练模型时，需要提供训练数据、训练轮数、批量大小等参数。
        5. 评估模型：使用 `model.evaluate()` 方法评估模型的性能。在评估模型时，需要提供测试数据和评估指标等。
    通过以上步骤，可以使用 `Sequential` 模型构建、训练和评估深度神经网络。
    '''
    model = Sequential()
    # 1. 原始图片resize到(224, 224, 3)

    # 2. 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    # 所建模型后输出为48特征层
    '''
    卷积：改变h,w 人为设置c
            (f,f)           h-f+2p           w-f+2p               224-11           224-11
    (h,w)-------------> ( ---------- + 1 , ---------- + 1)  =  ( --------- + 1, ---------- + 1) = (55, 55, 96)
           步长s,填充p          s                s                    4                4
    '''
    model.add(
        Conv2D(
            filters=48,               # 指定卷积层的滤波器数量为 48 (减半)
            kernel_size=(11,11),      # 设置卷积核的大小为 11x11
            strides=(4,4),            # 定义卷积的步长为 4x4
            padding='valid',          # 选择卷积的填充方式为"valid"，表示不进行填充
            input_shape=input_shape,  # 指定输入数据的形状 (224, 224, 3)
            activation='relu'         # 使用 ReLU 激活函数  f(x)=max(0,x)  能够有效地解决梯度消失问题，并且计算简单，能够提高神经网络的训练速度和性能。
        )
    )
    
    model.add(BatchNormalization())
    # 3. 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    '''
    池化：改变h,w 不改变c
            (f,f)           h-f+2p           w-f+2p               55 - 3           55 -3
    (h,w)-------------> ( ---------- + 1 , ---------- + 1)  =  ( --------- + 1, ---------- + 1) = (27, 27, 96)
           步长s,填充p          s                s                    2                2
    '''
    model.add(
        MaxPooling2D(
            pool_size=(3,3), 
            strides=(2,2), 
            padding='valid'
        )
    )
    # 4. 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128, 
            kernel_size=(5,5), 
            strides=(1,1), 
            padding='same',
            activation='relu'
        )
    )
    
    model.add(BatchNormalization())
    # 5. 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )
    # 6. 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192, 
            kernel_size=(3,3),
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    ) 
    # 7. 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192, 
            kernel_size=(3,3), 
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    )
    # 8. 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128, 
            kernel_size=(3,3), 
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    )
    # 9. 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
    model.add(
        MaxPooling2D(
            pool_size=(3,3), 
            strides=(2,2), 
            padding='valid'
        )
    )
    # 10. 两个全连接层，最后输出为1000类,这里改为2类
    # 缩减为1024
    '''
    展平：`model.add(Flatten())` 是在 Keras 中添加一个 Flatten 层。
        Flatten 层的作用是将输入的多维数据展平为一维数据。它将输入数据的形状从 `(batch_size, height, width, channels)` 
                                                             转换为 `(batch_size, height * width * channels)`。
        在卷积神经网络中，通常在卷积层之后使用 Flatten 层，将卷积层的输出展平为一维向量，以便后续可以连接全连接层进行分类或回归任务。
        
        第9步中如果卷积层的输出形状为 `(6,6,256))`，则经过 Flatten 层后，输出的形状将变为 `(9216)`。
        
        添加 Flatten 层可以使模型能够处理不同大小的输入图像，并将其转换为适合全连接层的一维向量表示。
    '''
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(output_shape, activation='softmax'))

    return model