import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, MaxPool2D, AveragePooling2D, \
    GlobalAveragePooling2D, Dense


def conv2d_bn(x_input, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    output = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=conv_name)(x_input)
    output = BatchNormalization(scale=False, name=bn_name)(output)
    output = Activation('relu', name=name)(output)
    return output


def inception_v3(input_shape=[299, 299, 3], classes=1000):
    # 299*299*3
    img_input = Input(input_shape)
    # output：149*149*32
    x = conv2d_bn(img_input, 32, (3, 3), (2, 2), padding='valid')
    # output：147*147*32
    x = conv2d_bn(x, 32, (3, 3), (1, 1), padding='valid')
    # output：147*147*64
    x = conv2d_bn(x, 64, (3, 3), (1, 1))
    # output：73*73*64
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    # output：73*73*80
    x = conv2d_bn(x, 80, (1, 1), strides=(1, 1), padding='valid')
    # output：71*71*192
    x = conv2d_bn(x, 192, (3, 3), strides=(1, 1), padding='valid')
    # output：35*35*192
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    # --------------------------------#
    #   Block1 35x35
    # --------------------------------#
    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, (1, 1))

    branch5x5 = conv2d_bn(x, 48, (1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, (5, 5))

    branch3x3dbl = conv2d_bn(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, (1, 1))
    # 1x1，5x5，3x3，pool分支拼接。64+64+96+32 = 256 nhwc-0123
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed0')

    # --------------------------------#
    #   Block2 35x35
    # --------------------------------#
    # Block1 part2
    # 35 x 35 x 256 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, (1, 1))

    branch5x5 = conv2d_bn(x, 48, (1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, (5, 5))

    branch3x3dbl = conv2d_bn(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, (1, 1))

    # 1x1，5x5，3x3，pool分支拼接。64+64+96+64 = 288 nhwc-0123
    layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed1')

    # --------------------------------#
    #   Block2 35x35
    # --------------------------------#
    # Block1 part3
    # 35 x 35 x 256 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, (1, 1))

    branch5x5 = conv2d_bn(x, 48, (1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, (5, 5))

    branch3x3dbl = conv2d_bn(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, (1, 1))

    # 1x1，5x5，3x3，pool分支拼接。64+64+96+64 = 288 nhwc-0123
    layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed2')

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, (3, 3), strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, (3, 3), strides=(2, 2), padding='valid')

    branch_pool = MaxPool2D((3, 3), strides=(2, 2))(x)
    # 3x3， 3x3dbl, pool分支拼接。384+96 = 480 nhwc-0123
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, (1, 1))

    branch7x7 = conv2d_bn(x, 128, (1, 1))
    # 使用1xn和nx1代替nxn的卷积
    branch7x7 = conv2d_bn(branch7x7, 128, (1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, (7, 1))

    # 两个7x7的卷积，第一个1x1的卷积是为了减少参数量
    branch7x7dbl = conv2d_bn(x, 128, (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (7, 1))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, (1, 1))
    # 分支相加：192+192+192+192 = 768
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed4')

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768

    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, (1, 1))

        branch7x7 = conv2d_bn(x, 160, (1, 1))
        # 使用1xn和nx1代替nxn的卷积
        branch7x7 = conv2d_bn(branch7x7, 160, (1, 7))
        branch7x7 = conv2d_bn(branch7x7, 192, (7, 1))

        # 两个7x7的卷积，第一个1x1的卷积是为了减少参数量
        branch7x7dbl = conv2d_bn(x, 160, (1, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (1, 7))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (7, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (1, 7))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (7, 1))

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, (1, 1))

        # 分支相加：192+192+192+192 = 768
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed' + str(5 + i))

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, (1, 1))

    branch7x7 = conv2d_bn(x, 192, (1, 1))
    # 使用1xn和nx1代替nxn的卷积
    branch7x7 = conv2d_bn(branch7x7, 192, (1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, (7, 1))

    # 两个7x7的卷积，第一个1x1的卷积是为了减少参数量
    branch7x7dbl = conv2d_bn(x, 192, (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (7, 1))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, (1, 1))

    # 分支相加：192+192+192+192 = 768
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed7')

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, (1, 1))
    branch3x3 = conv2d_bn(branch3x3, 320, (3, 3), strides=(2, 2), padding='valid')

    branch7x7 = conv2d_bn(x, 192, (1, 1))
    branch7x7 = conv2d_bn(branch7x7, 192, (1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, (7, 1))
    branch7x7 = conv2d_bn(branch7x7, 192, (3, 3), (2, 2), padding='valid')

    branch_pool = MaxPool2D((3, 3), strides=(2, 2))(x)
    # 分支相加：320+192 = ?
    x = layers.concatenate([branch3x3, branch7x7, branch_pool], axis=3, name='mixed8')

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, (1, 1))

        branch3x3 = conv2d_bn(x, 384, (1, 1))
        branch3x3_1 = conv2d_bn(branch3x3, 384, (1, 3))
        branch3x3_2 = conv2d_bn(branch3x3, 384, (3, 1))
        # out: 384 + 384 = 768
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, (1, 1))
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, (3, 3))
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, (1, 3))
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, (3, 1))
        # out: 384 + 384 = 768
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, (1, 1))
        # out: 320 + 786 + 768 + 192 = 2408
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed9' + str(9 + i))

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    inputs = img_input
    return keras.models.Model(inputs, x, name='inception_v3')


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def predict():
    model = inception_v3()
    model.summary()
    model.load_weights("./inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    img_path = 'elephant.jpg'
    image = keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    image_input = keras.preprocessing.image.img_to_array(image)
    image_input = tf.expand_dims(image_input, axis=0)
    image_input = preprocess_input(image_input)
    preds = model.predict(image_input)
    print('Predicted:', decode_predictions(preds))


if __name__ == '__main__':
    from utils import get_train_dataset

    batch_size = 200
    train_data, train_label = get_train_dataset("./train_data")
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam()
    model = inception_v3()
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    model.fit(train_data, train_label, batch_size=batch_size, epochs=100)
    model.save_weights("inception_traffic_light.h5")
