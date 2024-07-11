from keras import layers, Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten, \
    Dense, ZeroPadding2D


def BINK1(input_tensor, strides, filters, stage):
    conv_name = 'conv_bink1_stage' + str(stage)
    x = Conv2D(filters, (1, 1), strides=strides, name=conv_name + 'a')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same', strides=(1, 1), name=conv_name + 'b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * 4, (1, 1), strides=(1, 1), name=conv_name + 'c')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(filters * 4, (1, 1), strides=strides, name=conv_name + 'd')(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x


def BINK2(input_tensor, filters, stage, block):
    conv_name = 'conv_bink2_stage' + str(stage) + '_block' + str(block)
    x = Conv2D(int(filters / 4), (1, 1), name=conv_name + 'a')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(int(filters / 4), (3, 3), padding='same', name=conv_name + 'b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (1, 1), name=conv_name + 'c')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x


def stage0(input_tensor):
    x = Conv2D(64, (7, 7), strides=(2, 2))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x


def stage1(input_tensor):
    x = BINK1(input_tensor, 1, 64, 1)
    x = BINK2(x, 256, 1, 1)
    x = BINK2(x, 256, 1, 2)
    return x


def stage2(input_tensor):
    x = BINK1(input_tensor, 2, 128, 2)
    x = BINK2(x, 512, 2, 1)
    x = BINK2(x, 512, 2, 2)
    x = BINK2(x, 512, 2, 3)
    return x


def stage3(input_tensor):
    x = BINK1(input_tensor, 2, 256, 3)
    x = BINK2(x, 1024, 3, 1)
    x = BINK2(x, 1024, 3, 2)
    x = BINK2(x, 1024, 3, 3)
    x = BINK2(x, 1024, 3, 4)
    x = BINK2(x, 1024, 3, 5)
    return x


def stage4(input_tensor):
    x = BINK1(input_tensor, 2, 512, 4)
    x = BINK2(x, 2048, 4, 1)
    x = BINK2(x, 2048, 4, 2)
    return x


def new_model():
    input = Input(shape=[224, 224, 3])

    x = ZeroPadding2D((3, 3))(input)

    x = stage0(x)
    x = stage1(x)
    x = stage2(x)
    x = stage3(x)
    x = stage4(x)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(1000, activation='softmax', name='fc1000')(x)

    return Model(input, x, name='resnet50')


model = new_model()
model.summary()
