import tensorflow as tf

# 构建矩阵   常量
matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])

# 矩阵乘法
product = tf.matmul(matrix1,matrix2)

# 启动图
sess = tf.Session()
# 返回数组对象
result = sess.run(product)
print(result)
# 关闭会话
sess.close()

