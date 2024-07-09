import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

matrix1 = tf.constant([[1.,1.]])
matrix2 = tf.constant([[2.]])

product = tf.matmul(matrix2,matrix1)
with tf.Session() as sess:
    print(sess.run(product))
sess.close()





#(b,h)
# matrix1 = tf.constant([[1.,1.]])  # 1*2
# print(matrix1.shape)
# matrix2 = tf.constant([[2.],[2.]]) #2*1
# product = tf.matmul(matrix1,matrix2)
#
# sess = tf.Session()
# res = sess.run(product)
# print(res)
# sess.close()


