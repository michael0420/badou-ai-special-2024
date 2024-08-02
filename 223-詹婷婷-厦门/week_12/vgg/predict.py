import vgg16
import tensorflow as tf
import numpy as np
import utils

img1 = utils.load_image('H:/CV/PRE/pythonProject1/data/vgg16/elephant.jpg')
# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32, [None, None, 3])
resize_img = utils.resize_image(inputs, (224, 224))

#建立网络结构
prediction = vgg16.vgg_16(resize_img)

#载入模型
sess = tf.Session()
ckpt_filename = 'H:/CV/PRE/pythonProject1/data/vgg16/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

#最后进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs:img1})

#打印预测结果
print("result:")
utils.print_prob(pre[0], 'H:/CV/PRE/pythonProject1/data/vgg16/synset.txt')

