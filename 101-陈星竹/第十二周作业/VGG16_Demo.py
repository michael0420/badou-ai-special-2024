# 1.vgg16
import tensorflow as tf
import utils
slim = tf.contrib.slim # slim 是 TensorFlow 中的一个高层次 API，用于简化模型的定义和训练。

def vgg_16(inputs,num_classes = 1000,
           is_training=True, # 是否是训练模式 False则不经过dropout
           dropout_keep_prob=0.5, # dropout保留概率 默认0.5
           spatial_squeeze=True, # 是否压缩维度空间
           scope='vgg_16'):  # 变量作用域，默认是vgg16
    with tf.variable_scope(scope,'vgg_16',[inputs]):
        # 建立网络结构
        net = slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope='conv1')
        net = slim.max_pool2d(net,[2,2],scope='pool1')

        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2],scope='pool2')

        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2],scope='pool3')

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2],scope='pool4')

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2],scope='pool5') # 输出为（7，7，512）

        #利用卷积模拟全连接

        net = slim.conv2d(net,4096,[7,7],padding='VALID',scope='fc6')
        net = slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], padding='VALID', scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout7')

        # 如果希望在模型内定义softmax，在这里定义
        # 但是一般是在模型输出后通过单独的步骤完成，而不需要在模型定义中显式添加 softmax 激活。
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,normalizer_fn=None,scope='fc8')

        # 由于使用了卷积层模拟全连接层，所以输出需要进行平铺操作，
        # 即去除多余的维度，使输出变成一维张量。然后返回网络输出。
        if spatial_squeeze:
            net = tf.squeeze(net,[1,2],name='fc8/squeezed')
        return net

if __name__ == '__main__':
    # 读取图片
    img1 = utils.load_image("../cat.jpg")
    # 对输入的图片进行 resize，使其 shape 满足(-1,224,224,3)
    inputs = tf.placeholder(tf.float32, [None, None, 3])
    resized_img = utils.resize_image(inputs, (224, 224))

    # 建立网络结构
    prediction = vgg_16(resized_img)

    # 载入模型
    sess = tf.Session()
    ckpt_filename = '../model/vgg_16.ckpt'
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)

    # 最后结果进行 softmax 预测
    pro = tf.nn.softmax(prediction)
    pre = sess.run(pro, feed_dict={inputs: img1})

    # 打印预测结果
    print("result: ")
    utils.print_prob(pre[0], '../synset.txt')