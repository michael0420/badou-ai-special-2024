# -------------------------------------------------------------#
#   MobileNet的网络部分
# -------------------------------------------------------------#
import warnings
import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def MobileNet(input_shape=[224, 224, 3],  # 输入图像的形状大小为 224x224，有 3 个颜色通道（RGB）。
              depth_multiplier=1,  # 控制网络深度的参数。它乘以模型的基本深度，从而增加或减少网络的深度。
              dropout=1e-3,  # 用于防止过拟合的参数。它指定了在训练过程中要随机丢弃的神经元的比例
              classes=1000):  # 指定了要分类的类别数量, 模型将对 1000 个类别进行分类

    img_input = Input(shape=input_shape)

    # 224,224,3 -> 112,112,32  普通卷积（减小尺寸增加通道数）
    x = _conv_block(img_input, 32, strides=(2, 2))

    # 112,112,32 -> 112,112,64  深度可分离卷积（减少参数量）
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
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)  # 卷积代替全连接
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model


'''
定义了一个卷积块函数 `_conv_block`，用于在卷积神经网络中构建卷积层、批量归一化层和 ReLU 激活函数。
'''
def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):  # `inputs`：输入张量，通常是上一层的输出
    x = Conv2D(filters,  # 卷积核的数量，决定了输出特征图的深度
               kernel,  # 卷积核的大小，默认为 `(3, 3)`
               padding='same',  # 卷积操作会在输入的周围添加额外的零值，使得输出的特征图大小与输入的大小相同。这样做的目的是为了避免在卷积过程中丢失边缘信息，同时保持输出的空间维度与输入一致
               use_bias=False,  # 卷积层中不使用偏置项。这意味着卷积操作的输出仅取决于输入和卷积核的权重.
               strides=strides,  # `strides`：卷积的步长，默认为 `(1, 1)`
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)  # 对卷积层的输出进行批量归一化，以加速训练并提高模型的稳定性
    return Activation(relu6, name='conv1_relu')(x)  # 应用 ReLU 激活函数，将输入值限制在非负范围内，增加模型的非线性表达能力


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    # 深度可分离卷积
    '''
    定义深度可分离卷积块函数 `_depthwise_conv_block`，用于在深度学习模型中构建深度可分离卷积层。
        - `inputs`：输入张量，通常是图像或特征图。
        - `pointwise_conv_filters`：逐点卷积的滤波器数量，决定了输出特征图的深度。
        - `depth_multiplier`：深度乘数，用于控制深度可分离卷积的深度。
        - `strides`：卷积的步长，通常是一个元组表示 (高度步长, 宽度步长)。
        - `block_id`：块的标识符，用于命名层。
    深度可分离卷积块函数可以用于构建深度学习模型中的深度可分离卷积层，通过调整参数可以控制卷积核的大小、深度乘数、步长以及输出特征图的深度。
    深度可分离卷积是一种有效的卷积操作，可以减少模型的参数数量和计算量，同时提高模型的性能和效率。

    深度可分离卷积的尺寸和通道数可以通过以下方式进行改变：
        1. **尺寸**：深度可分离卷积的尺寸通常由卷积核的大小决定。你可以通过调整卷积核的高度和宽度来改变卷积的尺寸。例如，使用较大的卷积核将导致更大的感受野，从而捕捉更广泛的图像特征。
        2. **通道数**：深度可分离卷积的通道数可以通过以下几种方式进行调整：
            - **输入通道数**：改变输入图像或特征图的通道数。增加输入通道数将为卷积操作提供更多的信息。
            - **卷积核数量**：调整深度可分离卷积中的卷积核数量。增加卷积核数量将增加输出通道数，从而提取更多的特征。
            - **逐点卷积**：在深度可分离卷积之后，通常会使用逐点卷积来进一步调整通道数。通过改变逐点卷积的滤波器数量，可以增加或减少输出通道数。
        需要注意的是，改变深度可分离卷积的尺寸和通道数可能会对模型的性能和计算成本产生影响。在进行调整时，需要根据具体的任务和数据集进行实验和优化，以找到最适合的尺寸和通道数配置。
        此外，还可以考虑使用其他技术，如空洞卷积、分组卷积或增加卷积层数等，来进一步调整模型的特征提取能力和计算效率。这些技术可以与深度可分离卷积结合使用，以获得更好的模型性能。
    '''
    # 对输入进行深度可分离卷积操作，生成输出特征图
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    # 函数对输出进行批量归一化，以加速训练并提高模型的稳定性
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    # 函数对输出进行激活，通常使用 ReLU6 激活函数
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # 普通卷积
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):
    x /= 255.  # 将输入数据 x 除以 255，进行归一化处理，使得输入值在 0 到 1 之间
    x -= 0.5  # 将归一化后的输入数据 x 减去 0.5，进行中心化处理，使得输入值的均值为 0
    x *= 2.  # 将中心化后的输入数据 x 乘以 2，进行缩放处理，使得输入值的范围在 -1 到 1 之间
    return x


if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))

    img_path = 'elephant.jpg'
    # <PIL.Image.Image image mode=RGB size=224x224 at 0x2B65B0A70B8>
    img = image.load_img(img_path, target_size=(224, 224))
    '''
    ndarray=(224,224,3) [[[146. 149. 154.],  [147. 150. 155.],  [148. 151. 156.],  ...,  [130. 136. 148.], 
    '''
    x = image.img_to_array(img)
    '''
    ndarray=(1,224,224,3) [[[[146. 149. 154.],   [147. 150. 155.],   [148. 151. 156.],   ...,   [130. 136. 148.],  
    '''
    x = np.expand_dims(x, axis=0)
    '''
    ndarray=(1,224,224,3) [[[[ 0.14509809  0.1686275   0.20784318],   [ 0.15294123  0.17647064  0.21568632],   [ 0.16078436  0.18431377  0.22352946],   ...,   [ 0.0196079   0.06666672  0.16078436], 
    '''
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    '''
    ndarray=(1,1000)  [[1.50127313e-10 1.46900873e-12 4.05713463e-10 1.69053885e-11, 
    '''
    preds = model.predict(x)
    print(np.argmax(preds))  # 386
    print('Predicted:', decode_predictions(preds, 1))  # 只显示top1  Predicted: [[('n02504458', 'African_elephant', 0.7636121)]]
