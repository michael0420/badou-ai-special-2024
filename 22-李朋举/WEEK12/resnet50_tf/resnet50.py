#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
from __future__ import print_function
import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

'''
- `input_tensor`：input_tensor 是 identity_block 的输入 不是原始的输入
- `kernel_size`：卷积核的大小。
- `filters`：卷积核的数量，即输出的通道数。
- `stage`：表示当前是在网络的哪个阶段。
- `block`：表示当前是在阶段中的哪个块。
'''
#
def identity_block(input_tensor, kernel_size, filters, stage, block):

    """
    将变量 `filters` 中的值分别赋值给变量 `filters1`、`filters2` 和 `filters3`。
    这样的赋值操作常用于将一个数组或列表中的元素分别提取出来，以便在后续的代码中进行单独处理或使用。
    """
    filters1, filters2, filters3 = filters
    '''
    `'res' + str(stage) + block + '_branch'` 的意思是将字符串 `'res'`、变量 `stage` 转换为字符串后的值、变量 `block` 的值以及字符串 `'_branch'` 连接起来，形成一个新的字符串。
    用于给深度学习模型中的不同部分或层命名，以便在代码中更清晰地引用和操作它们。
    '''
    conv_name_base = 'res' + str(stage) + block + '_branch'
    '''
    `'bn' + str(stage) + block + '_branch'` 的意思是将字符串 `'bn'`、变量 `stage` 转换为字符串后的值、变量 `block` 的值以及字符串 `'_branch'` 连接起来，形成一个新的字符串。
    '''
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    '''
    定义一个卷积层、一个批量归一化层和一个激活函数，并将它们应用到输入张量 `input_tensor` 上。
    1. 使用 `Conv2D` 函数创建一个卷积层，该卷积层的卷积核大小为 `(1, 1)`，输出通道数为 `filters1`，并将其命名为 `conv_name_base + '2a'`。
    2. 使用 `BatchNormalization` 函数创建一个批量归一化层，并将其命名为 `bn_name_base + '2a'`。
    3. 使用 `Activation` 函数创建一个激活函数，该激活函数的类型为 `'relu'`，并将其应用到卷积层的输出上。
    这样，就完成了对输入张量的卷积、批量归一化和激活操作。
    '''
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


'''
定义一个卷积块（conv_block）
- `input_tensor`：输入张量，通常是前一层的输出。
- `kernel_size`：卷积核的大小。
- `filters`：卷积核的数量，即输出的通道数。
- `stage`：表示当前是在网络的哪个阶段。
- `block`：表示当前是在阶段中的哪个块。
- `strides`：卷积的步长。
在函数内部，可能会进行一系列的卷积操作、批量归一化操作和激活函数操作，以实现对输入张量的特征提取和转换。
卷积块的作用是通过卷积操作对输入特征进行降维或特征提取，同时通过批量归一化和激活函数操作增加模型的非线性表达能力。
'''
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters  # 将滤波器数量分解为三个部分

    conv_name_base = 'res' + str(stage) + block + '_branch'  # 创建一个卷积名称的基础字符串
    bn_name_base = 'bn' + str(stage) + block + '_branch'  # 创建一个批量归一化名称的基础字符串

    # 分支1

    x = Conv2D(filters1, (1, 1), strides=strides,  name=conv_name_base + '2a')(input_tensor)  # 使用 1x1 卷积核进行降维，并设置步长。
    x = BatchNormalization(name=bn_name_base + '2a')(x)  # 进行批量归一化
    x = Activation('relu')(x)  # 使用 ReLU 激活函数

    x = Conv2D(filters2, kernel_size, padding='same',name=conv_name_base + '2b')(x)  # 使用指定的卷积核大小进行卷积，并使用相同的填充。
    x = BatchNormalization(name=bn_name_base + '2b')(x)  # 进行批量归一化
    x = Activation('relu')(x)  # 使用 ReLU 激活函数  # 进行批量归一化

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)  # 使用 1x1 卷积核进行升维
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 分支2
    shortcut = Conv2D(filters3, (1, 1), strides=strides,name=conv_name_base + '1')(input_tensor)  # 使用 1x1 卷积核进行降维，并设置步长。
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)  # 进行批量归一化

    # 分支1 + 分支2  （add: 相同矩阵才能相加  concat: 拼接 h,w,c  在某个维度拼接其余两个维度需要相同才可以拼接）
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)  # 使用 ReLU 激活函数
    return x


'''
ResNet50:
   input -> Zeropad -> Conv2D -> BN -> ReLU -> MaxPool 
         ->  Conv Block - Identity Block - Identity Block 
         ->  Conv Block - Identity Block - Identity Block - Identity Block 
         ->  Conv Block - Identity Block - Identity Block - Identity Block - Identity Block - Identity Block
         ->  Conv Block - Identity Block - Identity Block
         ->  AveragePooling2D
         -> FC 
         -> out
'''
def ResNet50(input_shape=[224,224,3],classes=1000):

    """
    `Input()` 函数是 Keras 中的一个函数，用于定义模型的输入层。它接受一些参数来指定输入的形状和其他属性。
    `shape=input_shape` 表示输入的形状将与之前定义的 `input_shape` 相同。`input_shape` 通常是一个三元组，例如 `(224, 224, 3)`，表示图像的高度、宽度和通道数。
    通过定义输入层，我们可以将图像数据传递给后续的神经网络层进行处理和分析。
    """
    img_input = Input(shape=input_shape)
    '''
    使用Keras 中的 `ZeroPadding2D` 层对输入图像 `img_input` 进行了零填充。
    `ZeroPadding2D` 层的作用是在输入图像的周围添加一定数量的零像素，从而增加图像的大小。在这个例子中，`(3, 3)` 表示在图像的高度和宽度方向上都添加 3 个零像素。
    零填充可以在一定程度上防止卷积操作导致的图像尺寸减小，同时也可以帮助模型更好地学习图像的边界信息。
    '''
    x = ZeroPadding2D((3, 3))(img_input)

    '''
    定义了一个卷积神经网络（CNN）的一部分，其中包含了卷积层、批量归一化层、激活函数和最大池化层。    
    1. `x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)`: 这行代码定义了一个卷积层。`Conv2D` 函数用于创建卷积层，它接受几个参数：
        - `64`：表示卷积核的数量，即输出的特征图数量。
        - `(7, 7)`：表示卷积核的大小。
        - `strides=(2, 2)`：表示卷积的步长。
        - `name='conv1'`：为该层命名，方便在模型中进行引用。
    2. `x = BatchNormalization(name='bn_conv1')(x)`: 这行代码添加了一个批量归一化层。批量归一化是一种常见的神经网络层，用于对每个小批量的数据进行标准化处理，以加速训练并提高模型的稳定性。
    3. `x = Activation('relu')(x)`: 这行代码应用了一个 ReLU 激活函数。ReLU 激活函数是一种常用的激活函数，它将输入值限制在 0 到正无穷之间，有助于缓解神经网络中的梯度消失问题。
    4. `x = MaxPooling2D((3, 3), strides=(2, 2))(x)`: 这行代码定义了一个最大池化层。最大池化层用于对特征图进行下采样，通过选择每个区域内的最大值来减少特征图的尺寸，从而降低计算量并提取主要特征。
    综上所述，这段代码定义了一个包含卷积、批量归一化、ReLU 激活和最大池化的神经网络层序列。这样的层序列在 CNN 中常用于提取图像的特征，并逐渐减小特征图的尺寸，以便后续的处理。
    '''
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Conv2d Block 并联(分支1+分支2)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # Identity Block 串联
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
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

    '''
    对输入数据 `x` 进行了平均池化操作：
    `AveragePooling2D` 层的作用是对输入数据进行下采样，通过计算每个区域内的平均值来减少数据的维度。在这个例子中，`(7, 7)` 表示池化区域的大小，即对输入数据进行 7x7 的平均池化。
    平均池化操作可以帮助模型降低对输入数据中细节的敏感度，提取更具代表性的特征。它常用于图像分类等任务中，以减少输入数据的维度并增加模型的鲁棒性。
    '''
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    '''
    使用了 Keras 中的 `Flatten` 层将输入数据 `x` 展平为一维向量:
    `Flatten` 层的作用是将输入数据的维度从多维转换为一维。它将输入数据中的每个样本展平为一个一维向量，以便后续的全连接层或其他层进行处理。
    展平操作通常在卷积神经网络（CNN）的最后一层或在需要将多维数据转换为一维向量的情况下使用。通过将数据展平为一维向量，可以方便地与全连接层或其他需要一维输入的层进行连接。
    '''
    x = Flatten()(x) # 使用了 Keras 中的 Flatten 层将输入数据 x 展平为一维向量
    '''
   定义了一个全连接层（Dense Layer），用于将输入数据与输出数据进行连接：
    1. `x = Dense(classes, activation='softmax', name='fc1000')(x)`: 这行代码创建了一个全连接层。
        - `classes` 表示输出的类别数。在这个例子中，`classes` 可能是你要预测的类别数量。
        - `activation='softmax'` 指定了激活函数为 Softmax 函数。Softmax 函数通常用于多类别分类问题，它将输入值转换为概率分布，表示每个类别出现的可能性。
        - `name='fc1000'` 为该层命名，方便在模型中进行引用。
    通过将全连接层应用于输入数据 `x`，模型可以学习输入数据与输出类别之间的复杂关系，并进行分类或回归任务。
    '''
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    '''
    创建了一个 Keras 模型，将输入图像 `img_input` 和经过一系列处理后的输出 `x` 关联起来，并为模型指定了一个名称 `resnet50`。
    在这个模型中，`img_input` 是输入层，它接收图像数据。`x` 是经过一系列卷积、池化、归一化等操作后的输出。通过将这两个部分连接起来，形成了一个完整的神经网络模型。    
    '''
    model = Model(img_input, x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model

if __name__ == '__main__':
    # 声明网络结构
    model = ResNet50()  # 构建 ResNet50 网络模型
    '''
    使用 `model.summary()` 来获取模型的结构和参数信息的摘要。
        1. 模型的层结构：显示模型中包含的各个层，如卷积层、全连接层等。
        2. 每层的参数数量：告诉你每个层的参数数量，例如卷积核的数量、权重的数量等。
        3. 模型的总参数数量：给出整个模型的参数数量。
        4. 输入和输出的形状：描述模型的输入和输出张量的形状。
    通过查看模型的摘要，你可以了解模型的架构、参数数量以及输入输出的要求。这对于理解模型的复杂性、检查模型的配置是否正确以及进行模型的调试和优化都非常有帮助。    
    '''
    model.summary()  # 使用 model.summary() 来获取模型的结构和参数信息的摘要

    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    '''
    将图像 `img` 转换为一个 NumPy 数组 `x`:
    `image.img_to_array` 是 `keras.preprocessing.image` 模块中的一个函数，它的作用是将图像数据转换为适合深度学习模型处理的数组形式。
    转换后的数组 `x` 通常具有以下特点：
    1. 数组的维度：`x` 是一个多维数组，其维度通常与图像的尺寸和颜色通道数相关。例如，如果图像是彩色的，可能有三个颜色通道（红、绿、蓝），那么 `x` 的维度可能是 `(height, width, 3)`。
    2. 数值范围：图像的像素值通常在 0 到 255 之间（对于 8 位图像），转换后的数组 `x` 的数值范围也在 0 到 255 之间。
    3. 数据类型：`x` 的数据类型通常是 `numpy.uint8` 或类似的整数类型。
    在后续的代码中，你可以对转换后的数组 `x` 进行各种操作，例如数据预处理、特征提取、模型训练等。
    '''
    x = image.img_to_array(img)  # ndarray(224，224，3)
    '''
    使用 NumPy 库的 `expand_dims` 函数在数组 `x` 的指定轴（axis）上添加一个新的维度。
    在这个例子中，`axis=0` 表示在数组的第一个维度（通常是行维度）上添加一个新的维度。
    执行这行代码后，数组 `x` 的维度将增加 1，例如，如果 `x` 原本是一个二维数组，那么现在它将变成一个三维数组。
    添加新的维度通常是为了满足某些操作或算法的要求，例如在深度学习中，模型可能需要输入具有特定维度的张量。通过使用 `expand_dims` 函数，我们可以方便地在数组上添加或删除维度，以适应不同的计算需求。
    '''
    x = np.expand_dims(x, axis=0)  # ndarray(1,224，224，3)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))

    '''             编号                名称            概率                top5 
    Predicted: [ 
                [('n02504458', 'African_elephant', 0.76734424), 
                 ('n01871265', 'tusker', 0.19938569),
                 ('n02504013', 'Indian_elephant', 0.032160413),
                 ('n02410509', 'bison', 0.0005231188), 
                 ('n02408429', 'water_buffalo', 0.00030515814)]
                ]
    '''