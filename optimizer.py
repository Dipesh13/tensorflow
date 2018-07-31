# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

with open('./birth_life_2010.txt') as fi:
    data = fi.readlines()

lines = [d.split('\t') for d in data]
# print(lines)
features = [l[-2] for l in lines[1:]]
targets = [l[-1] for l in lines[1:]]
data = list(zip(features, targets))
data = np.asarray(data, dtype=np.float32)
# print(data)
n_samples = len(features)

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

w = tf.get_variable("w", initializer=tf.constant(0.0))
b = tf.get_variable("b", initializer=tf.constant(0.0))

y_pred = w * X + b

# keep variables same outside session eg) y_pred-y gives error , correct y_pred - Y .
loss = tf.square(y_pred - Y, name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_loss = 0
    for x, y in data:
        _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
        print(l)
        total_loss += l
    print("Avg loss {}".format(total_loss / n_samples))