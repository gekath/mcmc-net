'''
In this file we explore using tensorflow to classify 
hand-written digits using softmax, following the tutorial on
https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html
'''


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Create placeholder for input images, 784 is the input dimension
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# Our model, y is the predicted probability
y = tf.nn.softmax(tf.matmul(x,W) + b)
# True distribution, with one at the position of correct digit, and zero otherwise
y_ = tf.placeholder(tf.float32,[None,10])
# Our loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()
# Above creates a graph, now let's launch the graph in a session
sess = tf.Session()
sess.run(init)
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
sess.close()


