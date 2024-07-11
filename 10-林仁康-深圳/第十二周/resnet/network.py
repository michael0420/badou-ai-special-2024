import keras
from keras import layers, Input
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, AveragePooling2D, ZeroPadding2D
from keras.layers import Activation,BatchNormalization
from keras.models import Model

INPUT_SHAPE = (224,224,3)
OUTPUT_SHAPE = 2

class DebugInfo:
    stage = 0
    block = None

    def __init__(self, stage, block):
        self.stage = stage
        self.block = block

class Network:
    net = None

    # 横向学习
    def _ConvBlock(self, input, filter, step, debugInfo):
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
    def _IdentityBlock(self, input, debugInfo):
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

    def __init__(self):
        self.net = Sequential()

        # 初始化
        inputEle = Input(shape=INPUT_SHAPE)
        res = ZeroPadding2D((3, 3))(inputEle)

        # Stage 0
        # (224, 224, 3)
        res = Conv2D(64, (7, 7), strides=2, name='conv0_branch1')(res)
        res = BatchNormalization(name='bn_bantch1')(res)
        res = Activation('relu')(res)
        res = MaxPooling2D((3, 3), strides=2)(res)

        # Stage 1
        # 1 (56, 56, 64)
        res = self._ConvBlock(res, 64, 1, DebugInfo(1, '1'))
        # 2 (56, 56, 256)
        res = self._IdentityBlock(res, DebugInfo(1, '2'))
        # 3 (56, 56, 256)
        res = self._IdentityBlock(res, DebugInfo(1, '3'))

        # Stage 2
        # 1 (56, 56, 256)
        res = self._ConvBlock(res, 128, (2, 2), DebugInfo(2, '1'))
        # 2 (28, 28, 512)
        res = self._IdentityBlock(res, DebugInfo(2, '2'))
        # 3 (28, 28, 512)
        res = self._IdentityBlock(res, DebugInfo(2, '3'))
        # 4 (28, 28, 512)
        res = self._IdentityBlock(res, DebugInfo(2, '4'))

        # Stage 3
        # 1 (28, 28, 512)
        res = self._ConvBlock(res, 256, (2, 2), DebugInfo(3, '1'))
        # 2 (14, 14, 1024)
        res = self._IdentityBlock(res, DebugInfo(3, '2'))
        # 3 (14, 14, 1024)
        res = self._IdentityBlock(res, DebugInfo(3, '3'))
        # 4 (14, 14, 1024)
        res = self._IdentityBlock(res, DebugInfo(3, '4'))
        # 5 (14, 14, 1024)
        res = self._IdentityBlock(res, DebugInfo(3, '5'))
        # 6 (14, 14, 1024)
        res = self._IdentityBlock(res, DebugInfo(3, '6'))

        # Stage 4
        # 1 (14, 14, 1024)
        res = self._ConvBlock(res, 512, (2, 2), DebugInfo(4, '1'))
        # 2 (7, 7, 2048)
        res = self._IdentityBlock(res, DebugInfo(4, '2'))
        # 3 (7, 7, 2048)
        res = self._IdentityBlock(res, DebugInfo(4, '3'))

        # 最后池化
        res = AveragePooling2D((7, 7), name='avg_pool')(res)
        # 全连接
        res = Flatten()(res)
        res = Dense(OUTPUT_SHAPE, activation='relu', name="FC")(res)

        # 创建模型
        self.net = Model(inputEle, res, name='model')

if __name__ == "__main__":
    myNet = Network()