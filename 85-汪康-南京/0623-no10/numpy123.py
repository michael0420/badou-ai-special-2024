# -*- coding: utf-8 -*-
'''@Time: 2024/7/6 15:00

'''
import numpy
a = numpy.random.rand(3,3)-0.5
print(a)
import numpy as np
a = np.array([3, 1, 2, 4, 6, 1])
print(np.argmax(a))



# 创建两个数组
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# 使用 dot 方法计算点积
dot_product = np.dot(vector_a, vector_b)

print("点积:", dot_product,(1*4+2*5+3*6))
import tensorflow as tf
# 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3., 3.]])
# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1,matrix2)
sesson = tf.Session()
result = sesson.run(product)
print(result)
sesson.close()

state = tf.Variable(0,name='counter')
one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state,new_value)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print('state',sess.run(state))
    for _ in range(5):
        sess.run(update)
        print('update',sess.run(state))
