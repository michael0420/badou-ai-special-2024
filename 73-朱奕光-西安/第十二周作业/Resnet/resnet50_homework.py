import keras
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, name=conv_name_base + '2b', padding='same')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2c')(x)

    x = keras.layers.add([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, name=conv_name_base + '2b', padding='same')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2c')(x)

    y = keras.layers.Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    y = keras.layers.BatchNormalization(name=bn_name_base + '1')(y)

    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)
    return x

def ResNet50(input_shape=[224, 224, 3], classes=1000):
    img = keras.Input(shape=input_shape)
    x = keras.layers.ZeroPadding2D([3, 3])(img)

    x = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(x)
    x = keras.layers.BatchNormalization(name='bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], 2, 'a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], 2, 'b')
    x = identity_block(x, 3, [64, 64, 256], 2, 'c')

    x = conv_block(x, 3, [128, 128, 512], 3, 'a')
    x = identity_block(x, 3, [128, 128, 512], 3, 'b')
    x = identity_block(x, 3, [128, 128, 512], 3, 'c')
    x = identity_block(x, 3, [128, 128, 512], 3, 'd')

    x = conv_block(x, 3, [256, 256, 1024], 4, 'a')
    x = identity_block(x, 3, [256, 256, 1024], 4, 'b')
    x = identity_block(x, 3, [256, 256, 1024], 4, 'c')
    x = identity_block(x, 3, [256, 256, 1024], 4, 'd')
    x = identity_block(x, 3, [256, 256, 1024], 4, 'e')
    x = identity_block(x, 3, [256, 256, 1024], 4, 'f')

    x = conv_block(x, 3, [512, 512, 2048], 5, 'a')
    x = identity_block(x, 3, [512, 512, 2048], 5, 'b')
    x = identity_block(x, 3, [512, 512, 2048], 5, 'c')

    x = keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(classes, activation='relu', name='fc1000')(x)

    model = keras.models.Model(img, x, name='resnet50')
    model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    return model

if __name__ == '__main__':
    model = ResNet50()
    model.summary()
    img_path = 'bike.jpg'
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print(preds)
    preds = decode_predictions(preds)
    print(preds)