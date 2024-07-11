from nets import vgg16
import tensorflow as tf
import numpy as np
import utils

# 从指定路径读取图像，并返回处理后的图像（裁剪成中心正方形）。
img1 = utils.load_image("./test_data/dog.jpg")

# 定义一个占位符，用于接收输入图像任意高度和宽度的RGB图像。
inputs = tf.placeholder(tf.float32, [None, None, 3])

# 将输入的图像调整为224x224的大小，符合VGG16模型的输入要求。
resized_img = utils.resize_image(inputs, (224, 224))

# 构建VGG16模型，并传入调整大小后的图像。返回模型的预测结果。
prediction = vgg16.vgg_16(resized_img)

# 创建一个TensorFlow会话，用于运行计算图。
sess = tf.Session()

# 指定预训练模型的检查点文件路径。
ckpt_filename = './model/vgg_16.ckpt'

# 初始化全局变量。
sess.run(tf.global_variables_initializer())

# 创建一个saver对象，用于加载和保存模型。
saver = tf.train.Saver()

# 从指定的检查点文件中恢复模型权重。
saver.restore(sess, ckpt_filename)

# 对模型的预测结果应用Softmax函数，得到每个类别的概率。
pro = tf.nn.softmax(prediction)

# 运行会话，进行预测。将输入图像传递给占位符，并计算Softmax后的概率。
pre = sess.run(pro, feed_dict={inputs: img1})

# 打印结果的标识。
print("result: ")

# 打印预测结果的Top1和Top5类别及其概率。传入预测概率和类别标签文件路径。
utils.print_prob(pre[0], './synset.txt')
