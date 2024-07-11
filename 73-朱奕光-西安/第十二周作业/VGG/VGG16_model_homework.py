import tensorflow as tf

slim = tf.contrib.slim

def VGG16_model(inputs, is_training=True, spatial_squeeze=True, scope='vgg_16'):
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        """
        输入尺寸：224，224，3
        默认填充方式‘SAME’
        默认步长：1
        输出尺寸不变，通道数根据卷积核数量改变：224，224，64
        
        如填充方式为‘Valid’
        步长：2
        输出尺寸：(H-3)/2+1，(W-3)/2+1，64 = 111,111,64
        """
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        """
        输入尺寸：224，224，64
        池化窗口：2*2
        H(new) = (H-2) /2 +1 = 112
        W(new)一样
        输出尺寸：112，112，64
        """
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')   #输出112,112,128
        net = slim.max_pool2d(net, [2, 2], scope='pool2')                    #输出56,56,128
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')   #输出56,56,256
        net = slim.max_pool2d(net, [2, 2], scope='pool3')                    #输出28,28,256
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')   #输出28,28,512
        net = slim.max_pool2d(net, [2, 2], scope='pool4')                    #输出14,14,512
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')   #输出14,14,512
        net = slim.max_pool2d(net, [2, 2], scope='pool5')                    #输出7,7,512
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')   #卷积核代替全连接层，输出1,1,4096
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')                    #卷积核代替全连接层，输出1,1,4096
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout7')
        net = slim.conv2d(net, 1000, [1, 1],  activation_fn=None, normalizer_fn=None, scope='fc8')  #卷积核代替全连接层，输出1,1,1000，检测类型数量为1000，softmax函数在推理代码中

        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='squeeze')
        return net
