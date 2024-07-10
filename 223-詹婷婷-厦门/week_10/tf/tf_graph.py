import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

#创建op
product = tf.matmul(matrix1, matrix2)

#启动默认图
sess = tf.Session()
#调用sess的run()
res = sess.run(product)
print(res)
#关闭会话
sess.close()

