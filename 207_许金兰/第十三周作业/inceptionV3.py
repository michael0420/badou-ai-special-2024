import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image


def conv2d_bn(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    """
    对输入数据进行卷积、标准化、激活函数激活处理
    :param x:输入数据
    :param filters:卷积输出维度
    :param kernel_size:卷积核尺寸
    :param strides:步长
    :param padding:边缘填充模式
    :param name:名字
    :return:返回处理后的数据
    """
    if name is None:
        bn_name = None
        conv_name = None
    else:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(input_shape=[299, 299, 3], classes=1000):
    """
    inceptionV3模型
    :param input_shape: 输入数据维度
    :param classes: 分类类目数
    :return: 返回inception 模型
    """
    # 输入层
    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, (3, 3), strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, (3, 3), padding='valid')
    x = conv2d_bn(x, 64, (3, 3))
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, (1, 1), padding='valid')
    x = conv2d_bn(x, 192, (3, 3), padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # inception block 1 part 1
    branch1x1 = conv2d_bn(x, 64, (1, 1))

    branch5x5 = conv2d_bn(x, 48, (1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, (5, 5))

    branch3x3dbl = conv2d_bn(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, (1, 1))

    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed0')

    # inception block 1 part 2
    branch1x1 = conv2d_bn(x, 64, (1, 1))

    branch5x5 = conv2d_bn(x, 48, (1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, (5, 5))

    branch3x3dbl = conv2d_bn(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, (1, 1))

    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed1')

    # inception block 1 part 3
    branch1x1 = conv2d_bn(x, 64, (1, 1))

    branch5x5 = conv2d_bn(x, 48, (1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, (5, 5))

    branch3x3dbl = conv2d_bn(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, (1, 1))

    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed2')

    # inception block 2 part 1
    branch3x3 = conv2d_bn(x, 384, (3, 3), strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3), strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # inception block 2 part 2
    branch1x1 = conv2d_bn(x, 192, (1, 1))

    branch7x7 = conv2d_bn(x, 128, (1, 1))
    branch7x7 = conv2d_bn(branch7x7, 128, (1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, (7, 1))

    branch7x7dbl = conv2d_bn(x, 128, (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, (1, 1))
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed4')

    # inception block 2 part 3 and part 4
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, (1, 1))

        branch7x7 = conv2d_bn(x, 160, (1, 1))
        branch7x7 = conv2d_bn(branch7x7, 160, (1, 7))
        branch7x7 = conv2d_bn(branch7x7, 192, (7, 1))

        branch7x7dbl = conv2d_bn(x, 160, (1, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (7, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (1, 7))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (7, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7))

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, (1, 1))
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed' + str(5 + i))

    # inception block 2 part 5
    branch1x1 = conv2d_bn(x, 192, (1, 1))

    branch7x7 = conv2d_bn(x, 192, (1, 1))
    branch7x7 = conv2d_bn(branch7x7, 192, (1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, (7, 1))

    branch7x7dbl = conv2d_bn(x, 192, (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, (1, 1))
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed7')

    # inception block 3 part 1
    branch3x3 = conv2d_bn(x, 192, (1, 1))
    branch3x3 = conv2d_bn(branch3x3, 320, (3, 3), strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, (1, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, (1, 7))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, (7, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, (3, 3), strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # inception block 3 part 2 and part 3
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, (1, 1))

        branch3x3 = conv2d_bn(x, 384, (1, 1))
        branch3x3_1 = conv2d_bn(branch3x3, 384, (1, 3))
        branch3x3_2 = conv2d_bn(branch3x3, 384, (3, 1))
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, (1, 1))
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, (3, 3))
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, (1, 3))
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, (3, 1))
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, (1, 1))
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed' + str(9 + i))
    # 平均池化后全连接。
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input

    model = Model(inputs, x, name='inception_v3')
    model.load_weights("inception_v3.h5")
    return model


def preprocess_input(x):
    """
    对输入数据进行标准化处理
    :param x: 输入数据
    :return: 返回标准化后的数据
    """
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    # 加载模型权重
    model = InceptionV3()
    # 读取图片
    img = image.load_img('elephant.jpg', target_size=(299, 299))
    x = image.img_to_array(img)
    # 增加一个维度
    x = np.expand_dims(x, axis=0)
    # 对数据进行归一化处理
    x = preprocess_input(x)
    # 模型预测
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
