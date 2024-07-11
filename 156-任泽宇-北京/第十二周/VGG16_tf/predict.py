import vgg16
import tensorflow as tf
import numpy as np
import utils

# 读取图片，并进行处理
img = utils.load_image('../../../../../data/VGG16-tensorflow-master/dog.jpg')
inputs = tf.placeholder(tf.float32, [None, None, 3])
resied_img = utils.resize_image(inputs, (224, 224))

# 建立网络结构
prediction = vgg16.vgg_16(resied_img)

# 载入模型
sess = tf.Session()
ckpt_filename = '../../../../../data/VGG16-tensorflow-master/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

# 预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs:img})

# 结果展示
print("result: ")
utils.print_prob(pre[0], '../../../../../data/VGG16-tensorflow-master/synset.txt')