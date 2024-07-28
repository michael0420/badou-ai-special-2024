#   MobileNet
import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def MobileNet(input_shape=[224,224,3],
              depth_multiplier=1,
              dropout=1e-3,
              classes=1000):
    """
    :param input_shape: 输入图像的形状
    :param depth_multiplier: 深度乘数
    :param dropout: dropout比例
    :param classes: 类别数
    :return:
    """

    img_input = Input(shape=input_shape)

    # 卷积块
    x = _conv_block(img_input, 32, strides=(2, 2))

    # 112,112,32 -> 112,112,64
    # 对输入的特征图进行卷积操作。64表示特征图的通道数，
    # depth_multiplier表示深度卷积的可乘性因子，
    # block_id用于标识当前卷积块的序号。
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)
    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)
    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4)
    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)
    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6)
    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)
    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)
    # 7,7,1024 -> 1,1,1024
    # 全局平均池化
    x = GlobalAveragePooling2D()(x)
    # 将输入的张量x重新调整形状未（1，1，1024），即通道数为1，高和宽都为1，深度为1024
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    # 随机丢弃一部分神经元，防止模型过拟合
    x = Dropout(dropout, name='dropout')(x)
    # 卷积层
    x = Conv2D(classes, (1, 1),padding='same', name='conv_preds')(x)
    # 激活函数
    x = Activation('softmax', name='act_softmax')(x)
    # 将输出张量x重新调整形状为（classes，）
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model

def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    """
    定义卷积块
    :param inputs: 输入数据
    :param filters: 过滤器数量
    :param kernel:  卷积核大小
    :param strides: 步长
    :return:
    """
    # 2D卷积层
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    # 批量归一化
    x = BatchNormalization(name='conv1_bn')(x)
    # 通过激活函数输出
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """
     定义深度卷积块
    :param inputs:输入数据
    :param pointwise_conv_filters: 点卷积核数量
    :param depth_multiplier: 深度倍数
    :param strides:步长
    :param block_id: 块编号
    :return:
    """
    # 实现二维深度卷积操作
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    # 批量归一化
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    # 通过激活函数输出
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # 实现点卷积操作
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    # 批量归一化
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    # 通过激活函数输出
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def relu6(x):
    # 返回输入值x的修正线性单元（ReLU）激活结果。
    # 如果x的值大于6，则将其限制为6；
    # 如果x小于0，则返回0；如果x在0到6之间，则返回其本身
    return K.relu(x, max_value=6)

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))

    img_path = 'elephant.jpg'
    # 加载图片
    img = image.load_img(img_path, target_size=(224, 224))
    # 图片转换为numpy数组
    x = image.img_to_array(img)
    # 添加一个维度
    x = np.expand_dims(x, axis=0)
    # 数据预处理
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    # 预测
    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds,1))  # 只显示top1

