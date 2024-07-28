import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
import cv2

#  slim是一个使构建，训练，评估神经网络变得简单的库。它可以消除原生tensorflow里面很多重复的模板性的代码，让代码更紧凑，更具备可读性
#引入slim
slim=tf.contrib.slim


def vgg_16(inputs,num=1000,is_training=True,dropout_keep_prob=0.5,scope='vgg_16'):  #num表示输出结点数
#  variable_scope类   用于定义创建变量（层）的操作的上下文管理器。
     with tf.variable_scope(scope,'vgg16',[inputs]):    #假设输入大小为224,，224的图片
          model=slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope='conv1')  #做两次卷积  输出层数为64  卷积核大小3,3，此时输出为224,224,64
          model=slim.max_pool2d(model,[2,2],scope='pool1')  #池化核大小2,2  步长默认为1   此时输出为112,112,64

          model=slim.repeat(model,2,slim.conv2d,128,[3,3],scope='conv2')  #做两次卷积  输出层数为128  卷积核大小3,3，此时输出为112,112,128
          model=slim.max_pool2d(model,[2,2],scope='pool2')  #池化核大小2,2  步长默认为1   此时输出为56,56,128

          model=slim.repeat(model,3,slim.conv2d,256,[3,3],scope='conv3')  #做3次卷积  输出层数为256  卷积核大小3,3，此时输出为56,56,256
          model=slim.max_pool2d(model,[2,2],scope='pool3')  #池化核大小2,2  步长默认为1   此时输出为28,28,256

          model=slim.repeat(model,3,slim.conv2d,512,[3,3],scope='conv4')  #做3次卷积  输出层数为512  卷积核大小3,3，此时输出为28,28,512
          model=slim.max_pool2d(model,[2,2],scope='pool4')  #池化核大小2,2  步长默认为1   此时输出为14,14,512

          model=slim.repeat(model,3,slim.conv2d,512,[3,3],scope='conv5')  #做3次卷积  输出层数为512  卷积核大小3,3，此时输出为14,14,512
          model=slim.max_pool2d(model,[2,2],scope='pool5')  #池化核大小2,2  步长默认为1   此时输出为7,7,512

          model=slim.conv2d(model,4096,[7,7],padding='VALID',scope='fc6')  #用卷积模拟全连接--卷积核大小跟输入大小一致  此时输出为1,1,4096
          model=slim.dropout(model,dropout_keep_prob,is_training=is_training,scope='dropout6')

          model=slim.conv2d(model,4096,[1,1],scope='fc7')   #用卷积模拟全连接,此时输出为1，1,4096
          model=slim.dropout(model,dropout_keep_prob,is_training=is_training,scope='dropout7')

          model=slim.conv2d(model,num,[1,1],scope='fc8')    #用卷积模拟全连接,此时输出为1，1,1000
          #tf.squeeze()函数用于从张量形状中移除大小为1的维度
          print(model.shape)
          if True:
              model=tf.squeeze(model,[1,2],name='fc9')   ## 由于用卷积的方式模拟全连接层，所以输出需要平铺
          print(model.shape)
          return  model


def load_imgs(path):   #将图片裁剪为正方形
    img=cv2.imread(path)
    short_edge=min(img.shape[:2])  #取h，w的最小值
    h=int((img.shape[0]-short_edge)/2)
    w=int((img.shape[1]-short_edge)/2)
    crop_img=img[h:h+short_edge,w:w+short_edge]
    return crop_img

#命名空间其实就是给几个变量包一层名字，方便变量管理。函数是：tf.name_scope
#tf.expand_dims(inputs,i) ---将shape为[height, width, channels]的图像 变为[1, height, width, channels]的张量，简单的说就是增加一个维度  i代表在哪个位置增加
#align_corners：布尔型参数，默认为False，为True时，输入张量和输出张量的四个角的像素点的中心是对齐的，保留四个角的像素值
def resize_imgs(img,size,method=tf.image.ResizeMethod.BILINEAR,align_corners=False):
    with tf.name_scope('resize_img'):
        img=tf.expand_dims(img,0)
        img=tf.image.resize_images(img,size,method,align_corners)
        img=tf.reshape(img,tf.stack([-1,size[0],size[1],3]))
        return  img


# strip() 方法用于移除字符串头尾指定的字符(默认为空格或换行符)或字符序列
def print_prob(prob,file_path):   #打印设置
    records=[line.strip() for line in open(file_path).readlines()]
    predict=np.argsort(prob)[::-1]  ##将概率从大到小排列的结果的序号存入predict
    top1=records[predict[0]]
    print('top1:',top1,prob[predict[0]])
    top5= [(records[predict[i]], prob[predict[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1

img=load_imgs('F:/PNG/jinmao.png')
# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs=tf.placeholder(tf.float32,[None,None,3])
resize_img=resize_imgs(inputs,(224,224))
print('reszie_img.shape',resize_img.shape)

#建立网络结构
pridiction=vgg_16(resize_img)

#载入训练好的模型
sess=tf.Session()
model_filepath='./vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()  #tf.train.Saver.save/restore--保存/加载模型
saver.restore(sess,model_filepath)

pro=tf.nn.softmax(pridiction)   #softmax不需要训练，因此可以放在外面
pre=sess.run(pro,feed_dict={inputs:img})

print('result:')
print_prob(pre[0],'F:/labels/labels.txt')






