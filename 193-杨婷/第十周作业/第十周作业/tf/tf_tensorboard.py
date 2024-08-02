import tensorflow as tf

# 定义一个计算图，实现两个向量的减法操作  
# 定义两个输入，a为常量，b为变量  
a = tf.constant([10.0, 20.0, 40.0], name='a')
b = tf.Variable(tf.random_uniform([3]), name='b')
'''
tf.add 用于两个张量的逐元素加法。
tf.add_n 用于将多个张量相加，这些张量必须具有兼容的形状或可以广播到相同的形状。
'''
output = tf.add_n([a, b], name='add')

# 生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日志中  
writer = tf.summary.FileWriter('logs', tf.get_default_graph())
writer.close()

# 启动tensorboard服务（在命令行启动）
# 将路径cd到logs文件夹的上一层（cd /d E:\八斗AI\第十周作业\tf）
# tensorboard --logdir=logs
# 启动tensorboard服务后，复制地址并在本地浏览器中打开，

# 查看tensorflow版本 python -c "import tensorflow as tf; print(tf.__version__)"   ----1.15
# 查看tensorboard版本 pip show tensorboard  ----1.15.0
# 查看tensorflow-estimator版本 pip show tensorflow-estimator  -----1.15.1
