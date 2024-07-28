from keras import Input, Model
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, AveragePooling2D, Dense


def new_model(num_classes, input_shape=None):
    if input_shape is None:
        input_shape = [224, 224, 3]
    img_input = Input(shape=input_shape)
    # 224 x 224 x 3 -> 112 x 112 x 32
    net = conv_block(img_input, 32, (3, 3), (2, 2))
    # 112 x 112 x 32 -> 112 x 112 x 32
    net = depthwise_conv_block(net, 32)
    # 112 x 112 x 32 -> 112 x 112 x 64
    net = conv_block(net, 64, (1, 1))
    # 112 x 112 x 64 -> 56 x 56 x 64
    net = depthwise_conv_block(net, 64, (2, 2))
    # 56 x 56 x 64 -> 56 x 56 x 128
    net = conv_block(net, 128, (1, 1))
    # 56 x 56 x 128 -> 56 x 56 x 128
    net = depthwise_conv_block(net, 128)
    # 56 x 56 x 128 -> 56 x 56 x 128
    net = conv_block(net, 128, (1, 1))
    # 56 x 56 x 128 -> 28 x 28 x 128
    net = depthwise_conv_block(net, 128, (2, 2))
    # 28 x 28 x 128 -> 28 x 28 x 256
    net = conv_block(net, 256, (1, 1))
    # 28 x 28 x 256 -> 28 x 28 x 256
    net = depthwise_conv_block(net, 256)
    # 28 x 28 x 256 -> 28 x 28 x 256
    net = conv_block(net, 256, (1, 1))
    # 28 x 28 x 256 -> 14 x 14 x 256
    net = depthwise_conv_block(net, 256, (2, 2))
    # 14 x 14 x 256 -> 14 x 14 x 512
    net = conv_block(net, 512, (1, 1))

    # 5 x 14 x 14 x 512
    for _ in range(5):
        # 14 x 14 x 512 -> 14 x 14 x 512
        net = depthwise_conv_block(net, 512)
        # 14 x 14 x 512 -> 14 x 14 x 512
        net = conv_block(net, 512, (1, 1))

    # 14 x 14 x 512 -> 7 x 7 x 512
    net = depthwise_conv_block(net, 512, (2, 2))
    # 7 x 7 x 512 -> 7 x 7 x 1024
    net = conv_block(net, 1024, (1, 1))
    # 7 x 7 x 1024 -> 7 x 7 x 1024
    net = depthwise_conv_block(net, 1024)
    # 7 x 7 x 1024 -> 7 x 7 x 1024
    net = conv_block(net, 1024, (1, 1))

    net = AveragePooling2D((7, 7))(net)
    net = Dense(num_classes, activation='softmax', name='classifier')(net)

    return Model(img_input, net, name='mobilenet')


def relu6(net):
    return K.relu(net, max_value=6)


def conv_block(img_input, filters, kernel, strides=(1, 1), padding='same'):
    img_input = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding)(img_input)
    img_input = BatchNormalization(scale=False)(img_input)
    img_input = Activation(relu6)(img_input)
    return img_input


def depthwise_conv_block(image_input, filters, strides=(1, 1)):
    image_input = DepthwiseConv2D(
        kernel_size=(3, 3),
        padding='same',
        depth_multiplier=1,
        strides=strides,
        use_bias=False)(image_input)
    image_input = BatchNormalization()(image_input)
    image_input = Activation(relu6)(image_input)
    image_input = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(image_input)
    image_input = BatchNormalization()(image_input)
    return Activation(relu6)(image_input)


if __name__ == '__main__':
    model = new_model(1000)
    model.summary()
