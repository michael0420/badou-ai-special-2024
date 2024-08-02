from keras import layers, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, TimeDistributed, AveragePooling2D, Add
from keras.layers import Activation,BatchNormalization
from keras import backend as K

INPUT_SHAPE = (600, 600, 3)
OUTPUT_SHAPE = 2

class DebugInfo:
    stage = 0
    block = None

    def __init__(self, stage, block):
        self.stage = stage
        self.block = block

# 横向学习
def _ConvBlock(input, filter, step, debugInfo):
    name = 'conv' + str(debugInfo.stage) + '_branch' + debugInfo.block
    bnName = 'bn' + str(debugInfo.stage) + '_branch' + debugInfo.block

    res1 = Conv2D(filter, (1, 1), strides=step, padding='SAME', name=name+'a')(input)
    res1 = BatchNormalization(name=bnName+'a')(res1)
    res1 = Activation('relu')(res1)

    res1 = Conv2D(filter, (3, 3), padding='SAME', name=name+'b')(res1)
    res1 = BatchNormalization(name=bnName+'b')(res1)
    res1 = Activation('relu')(res1)

    res1 = Conv2D(filter * 4, (1, 1), padding='SAME', name=name+'c')(res1)
    res1 = BatchNormalization(name=bnName+'c')(res1)

    res2 = Conv2D(filter * 4, (1, 1), strides=step, padding='SAME', name=name+'2a')(input)
    res2 = BatchNormalization(name=bnName + '2a')(res2)

    res = layers.add([res1, res2])
    res = Activation('relu')(res)
    return res

# 纵向学习
def _IdentityBlock(input, debugInfo):
    channel = int(input.shape[3])
    name = 'identity' + str(debugInfo.stage) + '_branch' + debugInfo.block
    bnName = 'bn' + str(debugInfo.stage) + '_branch' + debugInfo.block

    res1 = Conv2D(channel // 4, (1, 1), padding='SAME', name=name + 'a')(input)
    res1 = BatchNormalization(name=bnName + 'a')(res1)
    res1 = Activation('relu')(res1)

    res1 = Conv2D(channel // 4, (3, 3), padding='SAME', name=name + 'b')(res1)
    res1 = BatchNormalization(name=bnName + 'b')(res1)
    res1 = Activation('relu')(res1)

    res1 = Conv2D(channel, (1, 1), padding='SAME', name=name + 'c')(res1)
    res1 = BatchNormalization(name=bnName + 'c')(res1)

    res = layers.add([res1, input])
    res = Activation('relu')(res)
    return res

def ResNet50(inputEle):
    # 初始化
    res = ZeroPadding2D((3, 3))(inputEle)

    # Stage 0
    # (224, 224, 3)
    res = Conv2D(64, (7, 7), strides=2, name='conv0_branch1')(res)
    res = BatchNormalization(name='bn_bantch1')(res)
    res = Activation('relu')(res)
    res = MaxPooling2D((3, 3), strides=2)(res)

    # Stage 1
    # 1 (56, 56, 64)
    res = _ConvBlock(res, 64, 1, DebugInfo(1, '1'))
    # 2 (56, 56, 256)
    res = _IdentityBlock(res, DebugInfo(1, '2'))
    # 3 (56, 56, 256)
    res = _IdentityBlock(res, DebugInfo(1, '3'))

    # Stage 2
    # 1 (56, 56, 256)
    res = _ConvBlock(res, 128, (2, 2), DebugInfo(2, '1'))
    # 2 (28, 28, 512)
    res = _IdentityBlock(res, DebugInfo(2, '2'))
    # 3 (28, 28, 512)
    res = _IdentityBlock(res, DebugInfo(2, '3'))
    # 4 (28, 28, 512)
    res = _IdentityBlock(res, DebugInfo(2, '4'))

    # Stage 3
    # 1 (28, 28, 512)
    res = _ConvBlock(res, 256, (2, 2), DebugInfo(3, '1'))
    # 2 (14, 14, 1024)
    res = _IdentityBlock(res, DebugInfo(3, '2'))
    # 3 (14, 14, 1024)
    res = _IdentityBlock(res, DebugInfo(3, '3'))
    # 4 (14, 14, 1024)
    res = _IdentityBlock(res, DebugInfo(3, '4'))
    # 5 (14, 14, 1024)
    res = _IdentityBlock(res, DebugInfo(3, '5'))
    # 6 (14, 14, 1024)
    res = _IdentityBlock(res, DebugInfo(3, '6'))

    # # Stage 4
    # # 1 (14, 14, 1024)
    # res = _ConvBlock(res, 512, (2, 2), DebugInfo(4, '1'))
    # # 2 (7, 7, 2048)
    # res = _IdentityBlock(res, DebugInfo(4, '2'))
    # # 3 (7, 7, 2048)
    # res = _IdentityBlock(res, DebugInfo(4, '3'))

    return res

def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Conv2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    #旧版本使用image_dim_ordering()
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

def classifier_layers(x, input_shape, trainable=False):
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)
    return x

if __name__ == "__main__":
    inputs = Input(shape=INPUT_SHAPE)
    model = ResNet50(inputs)
    # classifier_layers(inputs)
