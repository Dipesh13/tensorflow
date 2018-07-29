import tensorflow as tf

a = tf.add(3, 5)

sess = tf.Session()
print(sess.run(a))
sess.close()

#or

with tf.Session() as sess:
    print(sess.run(a))