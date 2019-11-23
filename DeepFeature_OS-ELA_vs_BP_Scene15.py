#from os_elm import AUTO_ELM   # Huang's algorithm
from auto_elm import AUTO_ELM   # Our proposed algorithm

import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm # Jupyter notebook should use this
import matplotlib.pyplot as plt
import os
import cv2
import scipy.io as scipy_io
import time

def softmax(a):
	c = np.max(a, axis=-1).reshape(-1, 1)
	exp_a = np.exp(a - c)
	sum_exp_a = np.sum(exp_a, axis=-1).reshape(-1, 1)
	return exp_a / sum_exp_a

def sigmoid(x):
	'''input x should be a 1D array!'''
	for i in range(len(x)):
		x[i] = 1/(1+math.e**(-x[i]))
	return x

n_input_nodes = 4096
n_hidden_nodes = 51  # used to be 256
n_output_nodes = 4096

# ===========================================
# Prepare dataset
# ===========================================
DIR = '../DeepFeatures/Scene15/'

x_train = np.load(DIR+'train_x_deep_feature.npy')/100.
y_train = np.load(DIR+'train_y_labels.npy')
x_test = np.load(DIR+'test_x_deep_feature.npy')/100.
y_test = np.load(DIR+'test_y_labels.npy')

"""shuffle dataset"""
p = np.random.permutation(len(x_train))
x_train = x_train[p]
y_train = y_train[p]

# divide the training dataset into two datasets:
# (1) for the initial training phase
# (2) for the sequential training phase
# NOTE: the number of training samples for the initial training phase
# must be much greater than the number of the model's hidden nodes.
# here, we assign int(1.5 * n_hidden_nodes) training samples
# for the initial training phase.
#border = int(1.5 * n_hidden_nodes)
border = 512
x_train_init = x_train[:border]
x_train_seq = x_train[border:]

print('total initial: ', (border))
print('total sequential: ', len(x_train_seq))



#####################################
#			OS-ELA				 #
#####################################
accuracy_sum = 0
time_sum = 0
for PPAP in tqdm(range(10)):
	tf.reset_default_graph()
	# ===========================================
	# Instantiate os-elm
	# ===========================================
	auto_elm = AUTO_ELM(
		# the number of input nodes.
		n_input_nodes=n_input_nodes,
		# the number of hidden nodes.
		n_hidden_nodes=n_hidden_nodes,
		# the number of output nodes.
		n_output_nodes=n_output_nodes,
		# loss function.
		# the default value is 'mean_squared_error'.
		# for the other functions, we support
		# 'mean_absolute_error', 'categorical_crossentropy', and 'binary_crossentropy'.
		c_value = 2.0**(2),
		loss='mean_squared_error',
		# activation function applied to the hidden nodes.
		# the default value is 'sigmoid'.
		# for the other functions, we support 'linear' and 'tanh'.
		# NOTE: OS-ELM can apply an activation function only to the hidden nodes.
		activation='sin',
	)
	# ===========================================
	# Training
	# ===========================================
	# the initial training phase
	t1 = time.time()
	auto_elm.init_train(x_train_init, x_train_init)
	t2 = time.time()
	time_sum+=(t2-t1)

	# the sequential training phase
	batch_size = 1024

	t1 = time.time()
	for epoch in range(100):
		for i in range(0, len(x_train_seq), batch_size):
			x_batch = x_train_seq[i:i+batch_size]
			if len(x_batch) != batch_size:
				break
			auto_elm.seq_train(x_batch, x_batch)
	t2 = time.time()
	time_sum+=(t2-t1)

	"""classification"""
	x_train_encoded = auto_elm.encoding(x_train)
	x_test_encoded = auto_elm.encoding(x_test)

	X = tf.placeholder(tf.float32, [None, n_hidden_nodes])
	Y = tf.placeholder(tf.int64, [None])
	Y_ = tf.one_hot(indices=Y, depth=15) # one_hot labels: [N,M]

	fc1 = tf.layers.dense(inputs=X,units=512,activation=tf.nn.relu)
	fc2 = tf.layers.dense(inputs=fc1,units=512,activation=tf.nn.relu)
	out = tf.layers.dense(inputs=fc2,units=15,activation=None)

	loss = tf.losses.softmax_cross_entropy(logits=out,onehot_labels=Y_)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out,axis=1),tf.argmax(Y_,axis=1)),dtype=tf.float32))

	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	BATCH_SIZE = 128
	for epoch in range(100):
		i = 0
		while i < len(x_train_encoded):
			try:
				batch_x = x_train_encoded[i:i+BATCH_SIZE]
				batch_y = y_train[i:i+BATCH_SIZE]
			except:
				batch_x = x_train_encoded[i:]
				batch_y = y_train[i:]
			i+=BATCH_SIZE
			sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})

	"""evaluation"""
	avg_testing_acc = sess.run(accuracy, feed_dict={X: x_test_encoded, Y: y_test})
	print("Testing accuracy: ",avg_testing_acc)
	accuracy_sum+=avg_testing_acc

print("==========================================")
print("OS-ELA ===================================")
print("==========================================")
print("Average time: ", time_sum/10.0)
print("Average accuracy: ", accuracy_sum/10.0)
print("==========================================")



#####################################
#			BP-A				   #
#####################################
accuracy_sum = 0
time_sum = 0
for PPAP in tqdm(range(10)):
	tf.reset_default_graph()

	"""BP autoencoder"""
	X = tf.placeholder(tf.float32, [None, n_input_nodes])
	Y = tf.placeholder(tf.float32, [None, n_output_nodes])

	encoding_layer = tf.layers.dense(inputs=X,units=n_hidden_nodes,activation=tf.math.sin)
	Y_hat = tf.layers.dense(inputs=encoding_layer,units=n_output_nodes,activation=None)

	loss = tf.losses.mean_squared_error(labels=Y,predictions=Y_hat)

	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	t1 = time.time()
	BATCH_SIZE = 128
	for epoch in range(100):
		i = 0
		while i < len(x_train):
			try:
				batch_x = x_train[i:i+BATCH_SIZE]
				batch_y = x_train[i:i+BATCH_SIZE]
			except:
				batch_x = x_train[i:]
				batch_y = x_train[i:]
			i+=BATCH_SIZE
			sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})
	t2 = time.time()
	time_sum+=(t2-t1)

	"""classification"""
	x_train_encoded = sess.run(encoding_layer, feed_dict={X: x_train})
	x_test_encoded = sess.run(encoding_layer, feed_dict={X: x_test})

	X = tf.placeholder(tf.float32, [None, n_hidden_nodes])
	Y = tf.placeholder(tf.int64, [None])
	Y_ = tf.one_hot(indices=Y, depth=15) # one_hot labels: [N,M]

	fc1 = tf.layers.dense(inputs=X,units=512,activation=tf.nn.relu)
	fc2 = tf.layers.dense(inputs=fc1,units=512,activation=tf.nn.relu)
	out = tf.layers.dense(inputs=fc2,units=15,activation=None)

	loss = tf.losses.softmax_cross_entropy(logits=out,onehot_labels=Y_)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out,axis=1),tf.argmax(Y_,axis=1)),dtype=tf.float32))

	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	BATCH_SIZE = 128
	for epoch in range(100):
		i = 0
		while i < len(x_train_encoded):
			try:
				batch_x = x_train_encoded[i:i+BATCH_SIZE]
				batch_y = y_train[i:i+BATCH_SIZE]
			except:
				batch_x = x_train_encoded[i:]
				batch_y = y_train[i:]
			i+=BATCH_SIZE
			sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})

	"""evaluation"""
	avg_testing_acc = sess.run(accuracy, feed_dict={X: x_test_encoded, Y: y_test})
	print("Testing accuracy: ",avg_testing_acc)
	accuracy_sum+=avg_testing_acc

print("==========================================")
print("BP	 ===================================")
print("==========================================")
print("Average time: ", time_sum/10.0)
print("Average accuracy: ", accuracy_sum/10.0)
print("==========================================")