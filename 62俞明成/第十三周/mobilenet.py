import numpy as np
from PIL import Image
from keras import Input, Model
from keras._tf_keras.keras import backend as K
from keras.src.layers import Reshape, Dropout
from keras_preprocessing import image
from keras._tf_keras.keras.layers import Conv2D, BatchNormalization, Activation, \
    GlobalAveragePooling2D, Dense, DepthwiseConv2D, Flatten
from keras._tf_keras.keras.applications.imagenet_utils import decode_predictions


def mobilenet(inputs_shape=[224, 224, 3], classes=1000):
    img = Input(shape=inputs_shape)
    x = conv2d_bn(img, 32, (3, 3), strides=(2, 2))
    x = depthwise_conv_block(x, 64, block_id=1)
    x = depthwise_conv_block(x, 128, block_id=2, stride=(2, 2))
    x = depthwise_conv_block(x, 128, block_id=3)
    x = depthwise_conv_block(x, 256, block_id=4, stride=(2, 2))
    x = depthwise_conv_block(x, 256, block_id=5)
    x = depthwise_conv_block(x, 512, block_id=6, stride=(2, 2))

    x = depthwise_conv_block(x, 512, block_id=7)
    x = depthwise_conv_block(x, 512, block_id=8)
    x = depthwise_conv_block(x, 512, block_id=9)
    x = depthwise_conv_block(x, 512, block_id=10)
    x = depthwise_conv_block(x, 512, block_id=11)

    x = depthwise_conv_block(x, 1024, block_id=12, stride=(2, 2))
    x = depthwise_conv_block(x, 1024, block_id=13, stride=(2, 2))

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(1e-3, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    model = Model(img, x, name='mobilenet')

    return model


def conv2d_bn(input, out_channels, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(out_channels, kernel_size, strides, padding, use_bias=False, name='conv1')(input)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    return x


def depthwise_conv_block(inputs, output_channels, block_id=1, stride=(1, 1)):
    x = DepthwiseConv2D((3, 3), strides=stride, depth_multiplier=1, padding='same', name='conv_dw_%d' % block_id,
                        use_bias=False)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation('relu', name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(output_channels, (1, 1), use_bias=False, name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)
    return x


def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = mobilenet()
    model.load_weights('mobilenet_1_0_224_tf.h5')
    model.summary()
    img_path = 'elephant.jpg'
    img = Image.open(img_path)
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))  # 只显示top1
