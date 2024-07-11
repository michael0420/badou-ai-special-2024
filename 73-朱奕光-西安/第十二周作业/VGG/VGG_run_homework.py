from VGG16_model_homework import VGG16_model
import tensorflow as tf
import utils_homework as utils
import cv2

labels_path = 'synset.txt'
img = utils.read_image('./test_data/table.jpg')

inputs = tf.placeholder(tf.float32, [None, None, 3])
img_input = utils.resize_image(inputs, [224,224])

res = VGG16_model(img_input)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt_path = './model/vgg_16.ckpt'
saver = tf.train.Saver()
saver.restore(sess, ckpt_path)

pro = tf.nn.softmax(res)
res = sess.run(pro, feed_dict={inputs: img})

utils.print_top(res[0], labels_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('img', img)
cv2.waitKey(0)
