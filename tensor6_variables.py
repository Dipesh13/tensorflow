# -*- coding: utf-8 -*-
import tensorflow as tf

s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant(5))
a = tf.add(s,m)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #tf.add()c reates an operatin .That operation needs to be executed in a session to take effect.
    #print(a)
    print(sess.run(a))