from keras import Input, Model, layers
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D


def new_model(num_classes, input_shape=None):
    if input_shape is None:
        input_shape = [299, 299, 3]
    img_input = Input(shape=input_shape)
    net = conv_block(img_input, 32, (3, 3), (2, 2), padding='valid')
    net = conv_block(net, 32, (3, 3), padding='valid')
    net = conv_block(net, 64, (3, 3))
    net = MaxPooling2D((3, 3), strides=(2, 2))(net)
    net = conv_block(net, 80, (3, 3), padding='valid')
    net = conv_block(net, 192, (3, 3), (2, 2), padding='valid')
    net = conv_block(net, 288, (3, 3))

    net = inception_block_a1(net)
    net = inception_block_a2(net)
    net = inception_block_a3(net)

    net = inception_block_b1(net)
    net = inception_block_b2(net)
    net = inception_block_b34(net, 3)
    net = inception_block_b34(net, 4)
    net = inception_block_b5(net)

    net = inception_block_c1(net)
    net = inception_block_c23(net, 2)
    net = inception_block_c23(net, 3)

    net = GlobalAveragePooling2D(name='avg_pool')(net)
    net = Dense(num_classes, activation='softmax', name='predictions')(net)

    return Model(img_input, net, name='inceptionV3')


def conv_block(img_input, filters, kernel, strides=(1, 1), padding='same'):
    img_input = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding)(img_input)
    img_input = BatchNormalization(scale=False)(img_input)
    img_input = Activation('relu')(img_input)
    return img_input


def inception_block_a1(net):
    branch1 = conv_block(net, 64, (1, 1))

    branch2 = conv_block(net, 48, (1, 1))
    branch2 = conv_block(branch2, 64, (5, 5))

    branch3 = conv_block(net, 64, (1, 1))
    branch3 = conv_block(branch3, 96, (3, 3))
    branch3 = conv_block(branch3, 96, (3, 3))

    branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
    branch4 = conv_block(branch4, 32, (1, 1))

    return layers.concatenate(
        [branch1, branch2, branch3, branch4],
        axis=3,
        name='inception_block_a1')


def inception_block_a2(net):
    branch1 = conv_block(net, 64, (1, 1))

    branch2 = conv_block(net, 48, (1, 1))
    branch2 = conv_block(branch2, 64, (5, 5))

    branch3 = conv_block(net, 64, (1, 1))
    branch3 = conv_block(branch3, 96, (3, 3))
    branch3 = conv_block(branch3, 96, (3, 3))

    branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
    branch4 = conv_block(branch4, 64, (1, 1))

    return layers.concatenate(
        [branch1, branch2, branch3, branch4],
        axis=3,
        name='inception_block_a2')


def inception_block_a3(net):
    branch1 = conv_block(net, 64, (1, 1))

    branch2 = conv_block(net, 48, (1, 1))
    branch2 = conv_block(branch2, 64, (5, 5))

    branch3 = conv_block(net, 64, (1, 1))
    branch3 = conv_block(branch3, 96, (3, 3))
    branch3 = conv_block(branch3, 96, (3, 3))

    branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
    branch4 = conv_block(branch4, 64, (1, 1))

    return layers.concatenate(
        [branch1, branch2, branch3, branch4],
        axis=3,
        name='inception_block_a3')


def inception_block_b1(net):
    branch1 = conv_block(net, 384, (3, 3), (2, 2), padding='valid')

    branch2 = conv_block(net, 64, (1, 1))
    branch2 = conv_block(branch2, 96, (3, 3))
    branch2 = conv_block(branch2, 96, (3, 3), (2, 2), padding='valid')

    branch3 = MaxPooling2D((3, 3), strides=(2, 2))(net)

    return layers.concatenate(
        [branch1, branch2, branch3],
        axis=3,
        name='inception_block_b1')


def inception_block_b2(net):
    branch1 = conv_block(net, 192, (1, 1))

    branch2 = conv_block(net, 128, (1, 1))
    branch2 = conv_block(branch2, 128, (1, 7))
    branch2 = conv_block(branch2, 192, (7, 1))

    branch3 = conv_block(net, 128, (1, 1))
    branch3 = conv_block(branch3, 128, (7, 1))
    branch3 = conv_block(branch3, 128, (1, 7))
    branch3 = conv_block(branch3, 128, (7, 1))
    branch3 = conv_block(branch3, 192, (1, 7))

    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(net)
    branch4 = conv_block(branch4, 192, (1, 1))

    return layers.concatenate(
        [branch1, branch2, branch3, branch4],
        axis=3,
        name='inception_block_b2')


def inception_block_b34(net, idx):
    branch1 = conv_block(net, 192, (1, 1))

    branch2 = conv_block(net, 160, (1, 1))
    branch2 = conv_block(branch2, 160, (1, 7))
    branch2 = conv_block(branch2, 192, (7, 1))

    branch3 = conv_block(net, 160, (1, 1))
    branch3 = conv_block(branch3, 160, (7, 1))
    branch3 = conv_block(branch3, 160, (1, 7))
    branch3 = conv_block(branch3, 160, (7, 1))
    branch3 = conv_block(branch3, 192, (1, 7))

    branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
    branch4 = conv_block(branch4, 192, (1, 1))

    return layers.concatenate(
        [branch1, branch2, branch3, branch4],
        axis=3,
        name=f'inception_block_b{idx}')


def inception_block_b5(net):
    branch1 = conv_block(net, 192, (1, 1))

    branch2 = conv_block(net, 192, (1, 1))
    branch2 = conv_block(branch2, 192, (1, 7))
    branch2 = conv_block(branch2, 192, (7, 1))

    branch3 = conv_block(net, 192, (1, 1))
    branch3 = conv_block(branch3, 192, (7, 1))
    branch3 = conv_block(branch3, 192, (1, 7))
    branch3 = conv_block(branch3, 192, (7, 1))
    branch3 = conv_block(branch3, 192, (1, 7))

    branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
    branch4 = conv_block(branch4, 192, (1, 1))

    return layers.concatenate(
        [branch1, branch2, branch3, branch4],
        axis=3,
        name='inception_block_b5')


def inception_block_c1(net):
    branch1 = conv_block(net, 192, (1, 1))
    branch1 = conv_block(branch1, 320, (3, 3), (2, 2), padding='valid')

    branch2 = conv_block(net, 192, (1, 1))
    branch2 = conv_block(branch2, 192, (1, 7))
    branch2 = conv_block(branch2, 192, (7, 1))
    branch2 = conv_block(branch2, 192, (3, 3), (2, 2), padding='valid')

    branch3 = MaxPooling2D((3, 3), strides=(2, 2))(net)

    return layers.concatenate(
        [branch1, branch2, branch3],
        axis=3,
        name='inception_block_c1')


def inception_block_c23(net, idx):
    branch1 = conv_block(net, 320, (1, 1))

    branch2 = conv_block(net, 384, (1, 1))
    branch2_1 = conv_block(branch2, 384, (1, 3))
    branch2_2 = conv_block(branch2, 384, (3, 1))
    branch2 = layers.concatenate([branch2_1, branch2_2], axis=3, name=f'inception_block_c{idx}_branch2_{idx}')

    branch3 = conv_block(net, 448, (1, 1))
    branch3 = conv_block(branch3, 384, (3, 3))
    branch3_1 = conv_block(branch3, 384, (1, 3))
    branch3_2 = conv_block(branch3, 384, (3, 1))
    branch3 = layers.concatenate([branch3_1, branch3_2], axis=3, name=f'inception_block_c{idx}_branch3_{idx}')

    branch4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(net)
    branch4 = conv_block(branch4, 192, (1, 1))

    return layers.concatenate(
        [branch1, branch2, branch3, branch4],
        axis=3,
        name=f'inception_block_c{idx}')


if __name__ == '__main__':
    model = new_model(1000)
    model.summary()
