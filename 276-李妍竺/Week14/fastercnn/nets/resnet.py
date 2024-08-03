#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#

from __future__ import print_function

from keras import layers, initializers, regularizers
from keras import backend as K
from keras.layers import Add,Input,ZeroPadding2D,Conv2D,TimeDistributed
from keras.layers import Activation,MaxPooling2D,AveragePooling2D
from keras.engine import Layer, InputSpec
from keras.models import Model

class BatchNormalization(Layer):
    def __init__(self, epsilon=1e-3, axis=-1,
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):

        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.axis = axis
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        input_shape：输入张量的形状。
        shape：在给定轴上计算权重形状。
        self.gamma、self.beta、self.running_mean 和 self.running_std：这些都是批量规范化所需的参数。γ和β是可训练的参数，而运行平均数和运行标准偏差是在训练过程中计算的。
        self.add_weight：添加模型权重的方法。
        self.built = True：设置为true以表示模型已构建
        '''
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name),
                                     trainable=False)
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name),
                                    trainable=False)
        self.running_mean = self.add_weight(shape=shape, initializer='zero',
                                            name='{}_running_mean'.format(self.name),
                                            trainable=False)
        self.running_std = self.add_weight(shape=shape, initializer='one',
                                           name='{}_running_std'.format(self.name),
                                           trainable=False)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):
        '''
        x：输入张量。
        mask：掩码，虽然在这个实现中被忽略了。
        input_shape：输入张量的形状。
        reduction_axes：要减少的轴。
        broadcast_shape：用于广播的形状。
        K.batch_normalization：这是对
        Keras
        后端函数的调用，它执行实际的批量规范化。
         '''

        assert self.built, 'Layer must be built before being called'
        input_shape = K.int_shape(x)

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
            x_normed = K.batch_normalization(
                x, self.running_mean, self.running_std,
                self.beta, self.gamma,
                epsilon=self.epsilon)
        else:
            # need broadcasting
            broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
            broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            x_normed = K.batch_normalization(
                x, broadcast_running_mean, broadcast_running_std,
                broadcast_beta, broadcast_gamma,
                epsilon=self.epsilon)

        return x_normed
    # 该函数返回一个字典，其中包含该层的配置参数，以便以后可以重新构建该层。它包括epsilon、轴以及正则化器等参数。
    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None}
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

###################
#  主干特征提取网络  #
###################
# 卷积块: 用于改变网络维度，因输入输出维度不一，所以不能连续串联
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters  # 通道

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 通道1
    x = Conv2D(filters1, (1, 1), strides=strides,name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
    x = BatchNormalization(name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3,(1, 1),name=conv_name_base+'2c')(x)
    x = BatchNormalization(name=bn_name_base+'2c')(x)

    # 通道2
    shortcut = Conv2D(filters3,(1, 1),strides=strides,name=conv_name_base+'1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base+'1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

# 恒等块: 用于串联网络,也称短路连接（快捷连接）
def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters  # filter参数

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

# Resnet50主网络结构
def ResNet50(inputs):
    img_input = inputs

    x = ZeroPadding2D(padding=(3, 3))(img_input)  # 四周添加0
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    return x


# 时间分布卷积层可以将一个卷积核分布在输入序列的每个时间步上，并在每个时间步上执行卷积操作。
# 这样，我们就可以在整个序列上共享相同的卷积核参数，并同时考虑序列中的所有时间步。
# 时间分布卷积块：构建一个时间分布卷积神经网络
def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Conv2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# 时间分布恒等块：构建一个时间分布卷积神经网络
def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

# 分类层
def classifier_layers(x, input_shape, trainable=False):
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    return x
if __name__ == "__main__":
    inputs = Input(shape=(600, 600, 3))
    model = ResNet50(inputs)
    model.summary()