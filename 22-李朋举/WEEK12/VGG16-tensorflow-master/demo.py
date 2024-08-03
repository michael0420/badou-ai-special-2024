from nets import vgg16
# import tensorflow as tf
import numpy as np
import utils

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 读取图片
img1 = utils.load_image("./test_data/dog.jpg")

# 对输入的图片进行resize，使其shape满足(-1,224,224,3=-0987654321`
inputs = tf.placeholder(tf.float32,[None,None,3])  # Tensor("Placeholder:0", shape=(?, ?, 3), dtype=float32)
resized_img = utils.resize_image(inputs, (224, 224))  # Tensor("resize_image/Reshape:0", shape=(1, 224, 224, 3), dtype=float32)

# 建立网络结构
prediction = vgg16.vgg_16(resized_img)

# 载入模型
sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
'''
`saver = tf.train.Saver()` 这行代码在 TensorFlow 中创建了一个 `Saver` 对象。`Saver` 对象用于保存和恢复 TensorFlow 模型的参数。
    - `tf.train.Saver()`：这是 TensorFlow 提供的用于创建 `Saver` 对象的函数。
    - `saver`：这是创建的 `Saver` 对象。
通过调用 `saver.save()` 方法，可以将模型的参数保存到指定的文件或路径中。在需要恢复模型时，可以使用 `saver.restore()` 方法从保存的文件中加载模型的参数。
此外，`Saver` 对象还提供了一些其他的功能，例如控制保存的频率、只保存部分参数等。你可以根据具体的需求来设置和使用 `Saver` 对象。
'''
saver = tf.train.Saver()
'''
`saver.restore(sess, ckpt_filename)` 是 TensorFlow 中用于恢复模型参数的函数。
    - `sess`：这是 TensorFlow 的会话对象，用于执行计算图。
    - `ckpt_filename`：这是要恢复的检查点文件的名称或路径。
该函数的作用是将之前保存的模型参数从检查点文件中加载到当前的会话中，以便继续训练或进行推理。
当执行 `saver.restore(sess, ckpt_filename)` 时，TensorFlow 会从指定的检查点文件中读取模型的参数，并将其赋值给相应的变量。这样，模型就可以从之前保存的状态继续运行。
需要注意的是，在执行恢复操作之前，需要确保已经创建了相应的模型结构和变量，并且与保存检查点时的模型结构完全一致。否则，可能会导致恢复失败或出现不匹配的情况。
此外，还需要确保检查点文件存在并且可以访问。如果检查点文件不存在或无法读取，将会抛出异常。
'''
saver.restore(sess, ckpt_filename)

# 最后结果进行softmax预测  Tensor("Softmax:0", shape=(1, 1000), dtype=float32)
pro = tf.nn.softmax(prediction)
# [[6.51762010e-13 4.96844796e-12 7.08291173e-13 2.43432965e-12,  4.76428862e-11 1.37680770e-11 9.13035598e-13 3.63949485e-13,  2.35147665e-12 2.17416620e-13 8.91054876e-14 1.34381080e-12,  4.90183957e-13 2.10277824e-12 1.00803873e-13 9.42346787e-14,  3.0939
pre = sess.run(pro,feed_dict={inputs:img1})

# 打印预测结果
print("result: ")
utils.print_prob(pre[0], './synset.txt')

'''
result: 
('Top1: ', 'n02099601 golden retriever', 0.9961261)
('Top5: ', [('n02099601 golden retriever', 0.9961261), ('n02099712 Labrador retriever', 0.0031311843), ('n02099267 flat-coated retriever', 0.00031133764), ('n02102480 Sussex spaniel', 0.00020649601), ('n02091831 Saluki, gazelle hound', 4.5306377e-05)])
'''
