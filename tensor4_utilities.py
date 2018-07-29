import tensorflow as tf
# Tensors filled with a specific value


input_tensor = [[0, 1], [2, 3], [4, 5]]
x = tf.zeros([2, 3], tf.int32) #[[0, 0, 0], [0, 0, 0]]

y = tf.zeros_like(input_tensor) # [[0, 0], [0, 0], [0, 0]]

z = tf.fill([2, 3], 8) # [[8, 8, 8], [8, 8, 8]]

# Constants as sequences
# tf.lin_space(start, stop, num, name=None)
a = tf.lin_space(10.0, 13.0, 4) # [10. 11. 12. 13.]

# tf.range(start, limit=None, delta=1, dtype=None, name='range')
b = tf.range(3, 18, 3) # [3 6 9 12 15]

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))
    print(sess.run(z))
    print(sess.run(a))
    print(sess.run(b))