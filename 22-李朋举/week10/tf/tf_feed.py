import tensorflow as tf

input1 = tf.placeholder(tf.float32)  # 使用placeholder占位（定义变量）
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # 以在session运行阶段，利用feed_dict的字典结构给placeholder填充具体的内容，而无需每次都提前定义好变量的值，大大提高了代码的利用率（填充变量）
    print(sess.run([output], feed_dict={input1: [7], input2: [2.]}))

# 输出:
# [array([ 14.], dtype=float32)]
