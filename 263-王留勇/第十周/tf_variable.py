import tensorflow as tf

state = tf.Variable(0, name="counter")

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_op)
	print('state', sess.run(state))
	for _ in range(5):
		sess.run(update)
		print("update", sess.run(state))