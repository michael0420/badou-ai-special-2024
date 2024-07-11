"""
AlexNet Keras implementation

"""

# Import necessary libs
import os
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Dense, Dropout, \
    Activation, Flatten, BatchNormalization, Input
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard, EarlyStopping
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"


def AlexNet(input_shape=(224, 224, 3), num_classes=10, l2_reg=0.0, weights=None):
    """
    AlexNet model
    :param input_shape: input shape
    :param num_classes: the number of classes
    :param l2_reg:
    :param weights:
    :return: model
    """
    input_layer = Input(shape=input_shape)

    # Layer 1
    # In order to get the same size of the paper mentioned, add padding layer first
    x = ZeroPadding2D(padding=(2, 2))(input_layer)
    x = conv_block(x, filters=96, kernel_size=(11, 11),
                   strides=(4, 4), padding="valid", l2_reg=l2_reg, name='Conv_1_96_11x11_4')
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_1_3x3_2")(x)

    # Layer 2
    x = conv_block(x, filters=256, kernel_size=(5, 5),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_2_256_5x5_1")
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_2_3x3_2")(x)

    # Layer 3
    x = conv_block(x, filters=384, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_3_384_3x3_1")

    # Layer 4
    x = conv_block(x, filters=384, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_4_384_3x3_1")

    # Layer 5
    x = conv_block(x, filters=256, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_5_256_3x3_1")
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_3_3x3_2")(x)

    # Layer 6
    x = Flatten()(x)
    x = Dense(units=4096)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 7
    x = Dense(units=4096)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 8
    x = Dense(units=num_classes)(x)
    x = BatchNormalization()(x)
    x = Activation("softmax")(x)

    if weights is not None:
        x.load_weights(weights)
    model = Model(input_layer, x, name="AlexNet")
    return model


def conv_block(layer, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', l2_reg=0.0, name=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_regularizer=l2(l2_reg),
               kernel_initializer="he_normal",
               name=name)(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


input_shape = (224, 224, 3)
num_classes = 10

alexnet = AlexNet(input_shape=input_shape, num_classes=num_classes)
alexnet.summary()

parallel_model = multi_gpu_model(alexnet, gpus=2)

epochs = 200
model_name = "AlexNet-2"
train_dir = r'/home/lst/datasets/cifar-10-images_train/'
test_dir = r'/home/lst/datasets/cifar-10-images_test/'
batch_size = 256
target_weight_height = (224, 224)

parallel_model.compile(loss=['categorical_crossentropy'],
                       optimizer='adadelta',
                       metrics=["accuracy"])
tensorboard = TensorBoard(log_dir=f'./logs/{model_name}', histogram_freq=0,
                          write_graph=True, write_images=False)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_weight_height,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_weight_height,
    batch_size=batch_size,
    class_mode='categorical')

num_train_samples = train_generator.samples
num_val_samples = validation_generator.samples

history = parallel_model.fit_generator(train_generator,
                                       validation_data=validation_generator,
                                       steps_per_epoch=math.ceil(num_train_samples / batch_size),
                                       validation_steps=math.ceil(num_val_samples / batch_size),
                                       epochs=epochs,
                                       callbacks=[tensorboard, early_stopping],
                                       )

parallel_model.save(f"{model_name}.h5")
