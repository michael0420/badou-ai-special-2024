# -------------------------------------------------------------#
#   vgg16的网络部分
# -------------------------------------------------------------#
import tensorflow as tf

# 创建slim对象
'''
`slim` 是 TensorFlow 的一个高级 API，它提供了一系列方便的函数和类，用于定义、训练和评估深度学习模型。
        使用 `slim` 可以简化深度学习模型的构建和训练过程，它提供了一些常用的模型结构和训练策略，例如卷积神经网络、循环神经网络、自动微分等。
        同时，`slim` 还提供了一些工具函数，例如模型评估、模型可视化、数据增强等。
如果你正在使用 TensorFlow 1.x 版本，可以使用 `tf.contrib.slim` 来构建和训练深度学习模型。( 2.x 版本中，`slim` 已经被移到了 `tf.keras` 中)
如果你正在使用 TensorFlow 2.x 版本，可以使用 `tf.keras` 来构建和训练深度学习模型，`tf.keras` 提供了更加简单和灵活的 API，同时也兼容 `tf.contrib.slim` 的一些功能。
'''
slim = tf.contrib.slim


def vgg_16(inputs,  # 输入张量，通常是图像数据
           num_classes=1000,  # 类别数，即模型要预测的类别数量
           is_training=True,  # 训练模式或测试模式的标志
           dropout_keep_prob=0.5,  # 在训练时，Dropout 层的保留概率
           spatial_squeeze=True,  # 是否在输出时挤压空间维度
           scope='vgg_16'):  # 模型的命名空间

    """
    创建了一个名为 scope 的变量作用域，其中 'vgg_16' 是作用域的名称，[inputs] 是作用域内的变量列表。在这个作用域内，所有的变量都将以 vgg_16 为前缀进行命名。 vgg_16/vgg_16/...
    scope 变量可能被用于控制 VGG16 模型的不同部分，例如卷积层、池化层或全连接层。通过将这些部分放在不同的作用域中，可以方便地管理和访问它们的变量。
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        # 建立vgg_16的网络

        # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
        '''
        在输入张量上应用两个卷积层，每个卷积层使用 64 个大小为 3x3 的卷积核。通过重复这个卷积操作，可以提取输入图像的特征:
            inputs：这是输入张量，通常是图像数据。     2：表示要重复的次数。      slim.conv2d：这是一个卷积层操作，用于对输入进行卷积运算。
            64：表示卷积核的数量（也称为输出通道数）。  [3, 3]：表示卷积核的大小。 scope='conv1'：为这个操作设置的作用域名称。
        '''
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 2X2最大池化，输出net为(112,112,64)
        '''
        使用TensorFlow 的 `slim` 库中的 `max_pool2d` 函数来执行最大池化操作。
            - `net`：这是输入的张量，通常是卷积层的输出。
            - `[2, 2]`：这是池化窗口的大小，表示在高度和宽度方向上分别进行 2 个像素的池化。
            - `scope='pool1'`：这是为池化操作设置的作用域名称。
        最大池化操作的作用是对输入张量进行下采样，通过选择池化窗口内的最大值来减少数据的维度，同时保留重要的特征信息。这样可以降低计算量、减少过拟合的风险，并增加模型的鲁棒性。
        在这个例子中，池化窗口的大小为 2x2，意味着输入张量的高度和宽度将分别减少一半。池化操作会在输入张量的每个通道上独立进行。
        通过设置作用域名称，可以更好地组织和管理模型的结构，方便在后续的代码中引用和操作这个池化层。
        '''
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 2X2最大池化，输出net为(56,56,128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 2X2最大池化，输出net为(28,28,256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(28,28,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # 2X2最大池化，输出net为(14,14,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(14,14,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # 2X2最大池化，输出net为(7,7,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')

        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net
