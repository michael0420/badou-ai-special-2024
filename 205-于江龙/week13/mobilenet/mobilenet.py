import numpy as np

from keras import layers, models
import keras

def relu6(x):
    return keras.activations.relu(x, max_value=6)

def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = layers.Conv2D(filters, kernel, padding='same', use_bias=False, 
                      strides=strides, name='conv1')(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    return layers.Activation(relu6, name='conv1_relu')(x)

def _depthwise_conv_block(inputs, pointwise_conv_filters, 
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    x = layers.DepthwiseConv2D((3, 3), padding='same',
                               depth_multiplier=depth_multiplier,
                               strides=strides, use_bias=False,
                               name='conv_dw_%d' % block_id)(inputs)
    x = layers.BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = layers.Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same', use_bias=False,
                      strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return layers.Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def MobileNet(input_shape=[224, 224, 3], depth_multiplier=1,
              dropout=1e-3, classes=1000):
    img_input = layers.Input(shape=input_shape)

    # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, 32, strides=(2, 2))

    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)

    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)

    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)

    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 1024), name='reshape_1')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    x = layers.Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = layers.Activation('softmax', name='act_softmax')(x)
    output = layers.Reshape((classes,), name='reshape_2')(x)

    model = models.Model(img_input, output, name='mobilnet_1_0_224_tf')
    return model

def process_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = MobileNet()
    model.load_weights('205-于江龙/week13/mobilenet/mobilenet_1_0_224_tf.h5')

    img = keras.preprocessing.image.load_img('205-于江龙/week13/elephant.jpg', target_size=(224, 224))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = process_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', keras.applications.imagenet_utils.decode_predictions(preds))