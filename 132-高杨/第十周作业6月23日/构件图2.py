import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

state = tf.Variable(0,name='counter')
one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state,new_value)
varible_init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(varible_init)
    print('改变前的值： ',sess.run(state))
    for _ in range(5):
        print(sess.run(update))








# 图变量的用法
# state = tf.Variable(0,name='counter')
#
# # 创建一个operation，作用是使state加一
# one = tf.constant(1)
# new_value = tf.add(state,one)
# update = tf.assign(state,new_value)
# #启动图后，变量需要先经过 初始化
# init_operation = tf.global_variables_initializer()
#
#
# with tf.Session() as sess:
#     sess.run(init_operation)
#     print('原来的值：',sess.run(state))
#     for _ in range(5):
#        sess.run(update)
#        print('变量赋值后的：',sess.run(state))
#



