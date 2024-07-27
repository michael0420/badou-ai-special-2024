import numpy as np
from keras import layers, models
import keras

def conv2d_bn(x, filters, kernel_size=(1,1), strides=(1,1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = layers.Conv2D(filters, kernel_size, 
                      strides=strides, 
                      padding=padding, 
                      use_bias=False, name=conv_name)(x)
    x = layers.BatchNormalization(scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x

def InceptionV3(input_shape=[299, 299, 3], classes=1000):
    img_input = layers.Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, kernel_size=(3, 3),strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, kernel_size=(3, 3), padding='valid')
    x = conv2d_bn(x, 64, kernel_size=(3, 3))
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, kernel_size=(1, 1), padding='valid')
    x = conv2d_bn(x, 192, kernel_size=(3, 3), padding='valid')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    #--------------------------------#
    #   Block1 35x35
    #--------------------------------#
    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, kernel_size=(1, 1))

    branch5x5 = conv2d_bn(x, 48, kernel_size=(1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, kernel_size=(5, 5))

    branch3x3dbl = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, kernel_size=(3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, kernel_size=(3, 3))

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, kernel_size=(1, 1))

    # 64+64+96+32 = 256  nhwc-0123
    x = layers.concatenate([
        branch1x1,
        branch5x5,
        branch3x3dbl,
        branch_pool
    ], axis=3, name='mixed0')

    # Block1 part2
    # 35 x 35 x 256 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, kernel_size=(1, 1))

    branch5x5 = conv2d_bn(x, 48, kernel_size=(1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, kernel_size=(5, 5))

    branch3x3dbl = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, kernel_size=(3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, kernel_size=(3, 3))

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, kernel_size=(1, 1))

    # 64+64+96+64 = 288
    x = layers.concatenate([
        branch1x1,
        branch5x5,
        branch3x3dbl,
        branch_pool
    ], axis=3, name='mixed1')

    # Block1 part3
    # 35 x 35 x 288 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, kernel_size=(1, 1))

    branch5x5 = conv2d_bn(x, 48, kernel_size=(1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, kernel_size=(5, 5))

    branch3x3dbl = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, kernel_size=(3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, kernel_size=(3, 3))

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, kernel_size=(1, 1))

    # 64+64+96+64 = 288
    x = layers.concatenate([
        branch1x1,
        branch5x5,
        branch3x3dbl,
        branch_pool
    ], axis=3, name='mixed2')

    #--------------------------------#
    #   Block2 17x17
    #--------------------------------#
    # Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384,  kernel_size=(3, 3), strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, kernel_size=(1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, kernel_size=(3, 3))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, kernel_size=(3, 3), strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([
        branch3x3,
        branch3x3dbl,
        branch_pool
    ], axis=3, name='mixed3')

    # Block2 part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, kernel_size=(1, 1))

    branch7x7 = conv2d_bn(x, 128, kernel_size=(1, 1))
    branch7x7 = conv2d_bn(branch7x7, 128, kernel_size=(1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, kernel_size=(7, 1))

    branch7x7dbl = conv2d_bn(x, 128, kernel_size=(1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, kernel_size=(1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(1, 7))

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, kernel_size=(1, 1))
    x = layers.concatenate([
        branch1x1,
        branch7x7,
        branch7x7dbl,
        branch_pool
    ], axis=3, name='mixed4')

    # Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, kernel_size=(1, 1))

        branch7x7 = conv2d_bn(x, 160, kernel_size=(1, 1))
        branch7x7 = conv2d_bn(branch7x7, 160, kernel_size=(1, 7))
        branch7x7 = conv2d_bn(branch7x7, 192, kernel_size=(7, 1))

        branch7x7dbl = conv2d_bn(x, 160, kernel_size=(1, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, kernel_size=(7, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, kernel_size=(1, 7))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, kernel_size=(7, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(1, 7))

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, kernel_size=(1, 1))
        x = layers.concatenate([
            branch1x1,
            branch7x7,
            branch7x7dbl,
            branch_pool
        ], axis=3, name='mixed' + str(5 + i))

    # Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, kernel_size=(1, 1))

    branch7x7 = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch7x7 = conv2d_bn(branch7x7, 192, kernel_size=(1, 7))
    branch7x7 = conv2d_bn(branch7x7, 192, kernel_size=(7, 1))

    branch7x7dbl = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(1, 7))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(7, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, kernel_size=(1, 7))

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, kernel_size=(1, 1))
    x = layers.concatenate([
        branch1x1,
        branch7x7,
        branch7x7dbl,
        branch_pool
    ], axis=3, name='mixed7')

    #--------------------------------#
    #   Block3 8x8
    #--------------------------------#
    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch3x3 = conv2d_bn(branch3x3, 320, kernel_size=(3, 3), strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, kernel_size=(1, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, kernel_size=(1, 7))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, kernel_size=(7, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, kernel_size=(3, 3), strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([
        branch3x3,
        branch7x7x3,
        branch_pool
    ], axis=3, name='mixed8')

    # Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, kernel_size=(1, 1))

        branch3x3 = conv2d_bn(x, 384, kernel_size=(1, 1))
        branch3x3_1 = conv2d_bn(branch3x3, 384, kernel_size=(1, 3))
        branch3x3_2 = conv2d_bn(branch3x3, 384, kernel_size=(3, 1))
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, kernel_size=(1, 1))
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, kernel_size=(3, 3))
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, kernel_size=(1, 3))
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, kernel_size=(3, 1))
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, kernel_size=(1, 1))
        x = layers.concatenate([
            branch1x1,
            branch3x3,
            branch3x3dbl,
            branch_pool
        ], axis=3, name='mixed' + str(9 + i))

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    model = models.Model(inputs, x, name='inception_v3')

    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = InceptionV3()
    model.load_weights('205-于江龙/week13/inceptionV3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

    img_path = '205-于江龙/week13/elephant.jpg'
    img = keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', keras.applications.imagenet_utils.decode_predictions(preds))


    