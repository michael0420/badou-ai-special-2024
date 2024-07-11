import tensorflow as tf

# 定义一个计算图，实现两个向量的减法操作
# 定义两个输入，a为常量，b为变量
a=tf.constant([10.0, 20.0, 40.0], name='a')
b=tf.Variable(tf.random_uniform([3]), name='b')
output=tf.add_n([a,b], name='add')

# 生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日志中
writer=tf.summary.FileWriter('logs', tf.get_default_graph())
writer.close()