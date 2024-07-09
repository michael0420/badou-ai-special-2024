import tensorflow as tf

# 构建矩阵   常量
num1 = tf.constant(3.0)
num2 = tf.constant(2.0)
num3 = tf.constant(5.0)

# 矩阵乘法
addNum = tf.add(num1,num2)
product = tf.multiply(num1,addNum)

with tf.Session() as session:
    result = session.run([addNum,product])   # 一次性运行多个op
    print(result)

