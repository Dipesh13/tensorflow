# create variables with tf.Variable
s = tf.Variable(2, name="scalar")

# create variables with tf.get_variable
s = tf.get_variable("scalar", initializer=tf.constant(2))