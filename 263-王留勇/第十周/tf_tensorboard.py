import tensorflow as tf

a = tf.constant([10.0, 20.0, 40.0], name='a')
b=tf.Variable(tf.random_uniform([3]), name='b')
output = tf.add_n([a, b], name='add')

writer = tf.summary.FileWriter('logs', tf.get_default_graph())
writer.close()