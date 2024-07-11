import tensorflow as tf

#创建一个变量，初始化为标量
state = tf.Variable(0, name="counter")

#创建一个op，使state加1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

#增加一个“初始化”op到图中
init_op = tf.global_variables_initializer()

#启动图，运行op
with tf.Session() as sess:
    #运行’init‘ op
    sess.run(init_op)
    #打印’state‘的初始值
    print("state", sess.run(state))
    #运行op，更新‘state’，并打印'state'
    for _ in range(5):
        sess.run(update)
        print("update:", sess.run(state))


