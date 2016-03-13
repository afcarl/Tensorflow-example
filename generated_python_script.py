#python
# Initialize
import sys
import os
import json
sys.argv = ['']
import tensorflow as tf
import numpy as np
learning_rate = 0.00005
epochs = 100000
# Read JSON files
with open('x_vals.json') as json_file:
  x_data = json.load(json_file)

with open('y_vals.json') as json_file:
  y_data = json.load(json_file)

# Run model
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in xrange(epochs):
  sess.run(train)
  if step % 10000 == 0:
    print(step, sess.run(W), sess.run(b))
    slope = sess.run(W)
    intercept = sess.run(b)

# Serialize results for importing back to R
with open('slope.json', 'w') as outfile:
  json.dump(slope.tolist(), outfile)

with open('intercept.json', 'w') as outfile:
  json.dump(intercept.tolist(), outfile)

