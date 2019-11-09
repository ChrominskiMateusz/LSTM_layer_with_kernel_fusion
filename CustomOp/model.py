# Based on:
# https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/

import tensorflow as tf

#import wrapper as wrap
from tensorflow.contrib import rnn
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

#unrolled through 28 time steps
time_steps = 28

#hidden LSTM units
num_units = 128

#rows of 28 pixels
input_size = 28

#learning rate for adam
learning_rate = 0.005

#mnist is meant to be classified in 10 classes(0-9).
n_classes = 10

#size of batch
batch_size = 128

#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random.normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random.normal([n_classes]))

#defining placeholders
#input image placeholder
x=tf.compat.v1.placeholder("float",[batch_size,time_steps,input_size])

#input label placeholder
y=tf.compat.v1.placeholder("float",[None,n_classes])

inputs = tf.transpose(x, [1, 0, 2])

#defining the network
fused_rnn_cell = rnn.LSTMBlockFusedCell(num_units)
outputs, _ = fused_rnn_cell(inputs, dtype=tf.float32)

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#initialize variables
init=tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    iter=1

    start_time = time.time()
    while iter<1000:
        batch_x,batch_y = mnist.train.next_batch(batch_size=batch_size)

        batch_x = batch_x.reshape((batch_size, time_steps, input_size))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x: batch_x, y:batch_y})
            los=sess.run(loss,feed_dict={x: batch_x, y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1

    duration = time.time() - start_time
    print(duration)