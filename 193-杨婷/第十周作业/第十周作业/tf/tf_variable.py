import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # 当前版本为2.x
# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name='counter')  # 计数器
# 创建一个 op, 其作用是使 state 增加 1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # 把new_value赋值给state（也可以直接赋值）

# 有变量，需要增加一个初始化op到图中
init_op = tf.global_variables_initializer()  # 来初始化图中所有全局变量的操作

with tf.Session() as sess:
    sess.run(init_op)
    print('state:', sess.run(state))
    for _ in range(5):
        sess.run(update)
        print('final state:', sess.run(state))
