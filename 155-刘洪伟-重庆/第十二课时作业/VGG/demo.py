# _*_ coding: UTF-8 _*_
# @Time: 2024/7/8 21:19
# @Author: iris
# @Email: liuhw0225@126.com
import utils
import tensorflow as tf
# from model import vggA16
import VGG16

if __name__ == '__main__':
    # 读取图片
    image = utils.load_image('./data/dog.jpg')

    inputs = tf.placeholder(tf.float32, [None, None, 3])
    # 进行resize/reshape操作。使其满足(-1, 224, 224, 3)
    image_new = utils.resize_iamge(inputs, (224, 224))

    # 建立网络结构
    network = VGG16.vgg_16(image_new)

    # 载入模型
    session = tf.Session()
    ckpt_filename = './model/vgg_16.ckpt'
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, ckpt_filename)

    # 最后结果进行softmax预测
    pro = tf.nn.softmax(network)
    pre = session.run(pro, feed_dict={inputs: image})

    # 打印预测结果
    print("result: ")
    utils.print_prob(pre[0], './synset.txt')
