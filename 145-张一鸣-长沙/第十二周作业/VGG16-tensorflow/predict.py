from nets import VGG16
import tensorflow as tf
import numpy as np
import utils

# 读取图片
img = utils.load_image('./test_data/table.jpg')

# 重设图片尺寸，使其shape满足(-1,224,224,3)
img_input = tf.placeholder(tf.float32, [None, None, 3])
img_resize = utils.resize_image(img_input, (224, 224))

# 定义VGG16网络
model = VGG16.VGG_16(img_resize)

# session.run()
ss = tf.Session()
file_name = './model/vgg_16.ckpt'
ss.run(tf.global_variables_initializer())
# 载入模型
tf.train.Saver().restore(ss, file_name)

# softmax
todo_softmax = tf.nn.softmax(model)
predict_result = ss.run(todo_softmax, feed_dict={img_input: img})

print('推理结果：')
utils.print_prob(predict_result[0], './synset.txt')
