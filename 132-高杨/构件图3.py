import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
input1 = tf.constant(1.0)
input2 = tf.constant(4.0)
input3 = tf.constant(6.0)
temp_res = tf.add(input2,input3)
final_res = tf.multiply(input1,temp_res)
with tf.Session() as sess:
    # session 可以抓取中间的结果

    print(sess.run([temp_res,final_res]))

