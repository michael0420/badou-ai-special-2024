import tensorflow as tf
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

intermediate = tf.add(input2, input3)  # 中间值
mul = tf.multiply(input1, intermediate)

with tf.Session() as sess:
    result = sess.run([intermediate, mul])  #需要获取的多个tensor值，在op的一次运行中一起获得（而不是逐个去获取 tensor）。
    print(result)
