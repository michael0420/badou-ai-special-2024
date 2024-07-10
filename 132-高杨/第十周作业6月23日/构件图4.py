import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# feed 用一个tensor临时替换一个操作得输出结果。可以提供feed数据作为run（）调用参数
# feed 只在调用它得方法内有效，方法结束，feed就会小时。最常见得用例是将某些特殊操作指定为feed
# 标记方法是使用 tf.placeholder()为这些操作创建占位符

# placeholder 是一个数据初始化得容器，与变量最大不同在于placeholder定义是一个模板
#这样我们就可以在session运行阶段，利用feed_dict得字典结构给placeholder填具体内容
#无须提前定义好变量值，大大提高了代码利用率

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
method = tf.matmul(input1,input2)
with tf.Session() as sess:
    print(sess.run([method],{input1:[[1.,2.]],input2:[[1.],[2.]]}))






# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.multiply(input1,input2)
# with tf.Session() as sess:
#     print(sess.run([output],feed_dict={input1:[7],input2:[2.]}))
#
#









