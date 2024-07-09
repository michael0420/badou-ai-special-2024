import tensorflow as tf
import numpy as np

# x = tf.constant(1.0)
# y = tf.constant(2.0)
# z1 = tf.add(x, y)
# z2 = tf.multiply(x, y)
#
# with tf.Session() as sess:
#     res = sess.run([z1, z2])
#     print(res)
#


# x = tf.constant(10)
# y = tf.constant(20)
# z = tf.multiply(x,y)
# with tf.Session() as sess:
#     res = sess.run(z)
#     print(res)
#


# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
# z = tf.multiply(x, y)
#
# with tf.Session() as sess:
#     # feed_dict 表示输入
#     res = sess.run(z, feed_dict={x: [10.0], y: [20.1]})
#     print(res)


# # 矩阵乘
# x = tf.constant([[1, 2]])
# y = tf.constant([[1], [2]])
# z = tf.matmul(x, y)
#
# # 固定步骤：必须创建Session会话，然后调用run，最后调用close
# sess = tf.Session()
# res = sess.run(z)
# print(res)
# sess.close()


# # 创建常量一维张量
# a = tf.constant([10.0, 20.0, 30.0], dtype=tf.float32, name='create_a')
# # 创建变量一维张量
# b = tf.Variable(tf.random_uniform([3]), dtype=tf.float32, name='create_b')
# c = tf.add(a, b, name='add')
# # 如果上面有定义了变量，那么一定要执行这句代码初始化全局的变量，并且sess.run(init)，否则会报错
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init) # 这句也必须要先执行，代表初始化全局变量
# res = sess.run(c)
# print(res)
# sess.close()
# writer = tf.summary.FileWriter('logs', tf.get_default_graph())
# writer.close()


# 创建一个变量
state = tf.Variable(0, name='count')
one = tf.constant(1)
new_val = tf.add(state, one, name='add')
# 把new_val赋值给state
# 在TensorFlow中，当你执行update = tf.assign(state, new_val)时，
# update代表一个表示赋值操作的对象，具体来说是一个tf.Operation对象。
# 这个对象在TensorFlow的计算图中表示一个节点，当这个节点被执行时，它
# 会将state变量的值更新为new_val的值。
# 简单来说就是定义update这个操作，在后面的run(update)的时候才会执行
update = tf.assign(state, new_val)

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        res = sess.run(update)
        print(f"update_{i} = {res}")


# n1 = np.array([[1], [2], [3], [4], [5]])
# n2 = np.array([[1, 2, 3, 4, 5]])
# n3 = np.dot(n1, n2)
# n4 = np.array([[1], [1], [1], [1], [1]])
# n3 = n3 + n4
# print(n3)
