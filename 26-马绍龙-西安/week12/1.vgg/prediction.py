import vgg16
import tensorflow as tf
import utils

# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_img = utils.resize_image(inputs, (224, 224))

# 建立网络结构
prediction = vgg16.vgg_16(resized_img)

# 载入模型
sess = tf.Session()
ckpt_filename = './vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()  # 创建一个Saver对象，用于保存和恢复模
saver.restore(sess, ckpt_filename)  # 创建一个Saver对象，用于保存和恢复模

# 最后结果进行softmax预测
pro = tf.nn.softmax(prediction)
img1 = utils.load_image("./test_data/dog.jpg")  # 读取图片
pre = sess.run(pro, feed_dict={inputs: img1})

# 打印预测结果
print("result: ")
utils.print_prob(pre[0], './synset.txt')
