import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


class MobileNet:
    def __init__(self, input_shape=[224, 224, 3], depth_multiplier=1, dropout=1e-3, classes=1000):
        """
        参数初始化
        :param input_shape: 输入数据形状
        :param depth_multiplier:
        :param dropout: dropout比例
        :param classes: 分类类目数
        """
        self.input_shape = input_shape
        self.depth_multiplier = depth_multiplier
        self.dropout = dropout
        self.classes = classes

    def conv2d_bn(self, inputs, filters, kernel=(3, 3), strides=(1, 1)):
        x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='conv1')(inputs)
        x = BatchNormalization(name='conv1_bn')(x)
        return Activation(self.relu6, name='conv1_relu')(x)

    def depthwiseconv_bn(self, inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
        x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides,
                            use_bias=False, name='conv_dw_%d' % block_id)(inputs)

        x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
        x = Activation(self.relu6, name='conv_dw_%d_relu' % block_id)(x)

        x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
                   name='conv_pw_%d' % block_id)(x)
        x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
        return Activation(self.relu6, name='conv_pw_%d_relu' % block_id)(x)

    def relu6(self, x):
        """
        设置激活函数
        :param x: 
        :return: 
        """
        return K.relu(x, max_value=6)

    def preprocess_input(self, x):
        """
        对输入数据进行归一化处理
        :param x: 输入数据
        :return: 返回归一化后的结果
        """
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def mobile_net(self):
        """
        构建mobilenet模型结构，并加载模型参数
        :return: 
        """
        # 输入层
        img_input = Input(shape=self.input_shape)
        # 卷积层
        x = self.conv2d_bn(img_input, 32, strides=(2, 2))
        x = self.depthwiseconv_bn(x, 64, self.depth_multiplier, block_id=1)
        x = self.depthwiseconv_bn(x, 128, self.depth_multiplier, strides=(2, 2), block_id=2)
        x = self.depthwiseconv_bn(x, 128, self.depth_multiplier, block_id=3)
        x = self.depthwiseconv_bn(x, 256, self.depth_multiplier, strides=(2, 2), block_id=4)
        x = self.depthwiseconv_bn(x, 256, self.depth_multiplier, block_id=5)
        x = self.depthwiseconv_bn(x, 512, self.depth_multiplier, strides=(2, 2), block_id=6)
        x = self.depthwiseconv_bn(x, 512, self.depth_multiplier, block_id=7)
        x = self.depthwiseconv_bn(x, 512, self.depth_multiplier, block_id=8)
        x = self.depthwiseconv_bn(x, 512, self.depth_multiplier, block_id=9)
        x = self.depthwiseconv_bn(x, 512, self.depth_multiplier, block_id=10)
        x = self.depthwiseconv_bn(x, 512, self.depth_multiplier, block_id=11)
        x = self.depthwiseconv_bn(x, 1024, self.depth_multiplier, strides=(2, 2), block_id=12)
        x = self.depthwiseconv_bn(x, 1024, self.depth_multiplier, block_id=13)

        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 1024), name='reshape_1')(x)
        x = Dropout(self.dropout, name='dropout')(x)
        x = Conv2D(self.classes, (1, 1), padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((self.classes,), name='reshape_2')(x)

        inputs = img_input
        model = Model(inputs, x, name='mobilenet_1_0_224_tf')
        model.load_weights('mobilenet_1_0_224_tf.h5')
        return model


if __name__ == '__main__':
    # 模型实例化
    model = MobileNet()
    # 读取图片
    img = image.load_img('elephant.jpg', target_size=(224, 224))
    # 图片处理
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = model.preprocess_input(x)
    # 模型预测
    preds = model.mobile_net().predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))
