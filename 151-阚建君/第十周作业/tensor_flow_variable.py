import tensorflow as tf

# 构建变量
state = tf.Variable(0, name='counter')
# 计数器每次加一
one = tf.constant(1)
# 构建op
new_value = tf.add(state,one)
update = tf.assign(state,new_value)
# 初始化
init_op = tf.global_variables_initializer()

# 启动图
with tf.Session() as sess:
    # 初始化
    sess.run(init_op)
    print("state的值为:%d" % (sess.run(state)))
    for i in range(5):
        sess.run(update)
        print("update的值为:%d" %(sess.run(state)))
