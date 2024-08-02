import tensorflow as tf

input1 = tf.placeholder(tf.float32)  # placeholder占位符，不确定数值只知道操作，先写上
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1: [7], input2: [2]}))  # 因为有placeholder所以第二个参数用feed_dict进行赋值
    