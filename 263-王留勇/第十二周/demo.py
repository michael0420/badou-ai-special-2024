import vgg16
import tensorflow as tf
import utils

img1 = utils.load_image("./test_data/dog.jpg")

inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_img = utils.resize_image(inputs, (224, 224))

prediction = vgg16.vgg_16(resized_img)

sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs:img1})

print('result:')
utils.print_prob(pre[0], './synset.txt')