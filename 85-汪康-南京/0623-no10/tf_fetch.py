# -*- coding: utf-8 -*-
'''@Time: 2024/7/10 20:19
fetch 抓取
feed 赋值
'''
import tensorflow as tf
a,b,c = tf.constant(3.0),tf.constant(2.0),tf.constant(5.0)
inter = tf.add(b,c)
mul = tf.multiply(a,inter)

with tf.Session() as sess:
    result = sess.run([mul,inter])
    print(result)

#feed，placeholder占位符，feed_dict用字典的形式赋值
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run([output],feed_dict={input1:[7],input2:[3.0]}))