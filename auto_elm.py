import numpy as np
#import tqdm
import tensorflow as tf
import os

"""This is our proposed algorithm!!"""

"""This is under testing"""
class AUTO_ELM(object):

	def __init__(
		self, n_input_nodes, n_hidden_nodes, n_output_nodes, c_value,
		activation='sigmoid', loss='mean_squared_error', name=None):

		if name == None:
			self.name = 'model'
		else:
			self.name = name

		self.__sess = tf.Session()
		self.__n_input_nodes = n_input_nodes             # m
		self.__n_hidden_nodes = n_hidden_nodes           # D
		self.__n_output_nodes = n_output_nodes           # m

		self.C = tf.constant(1.0*c_value)

		if activation == 'sigmoid':
			self.__activation = tf.nn.sigmoid
		elif activation == 'linear' or activation == None:
			self.__activation = tf.identity
		elif activation == 'tanh':
			self.__activation = tf.tanh
		elif activation == 'sin':
			self.__activation = tf.sin
		else:
			raise ValueError(
				'an unknown activation function \'%s\' was given.' % (activation)
			)

		if loss == 'mean_squared_error':
			self.__lossfun = tf.losses.mean_squared_error
		elif loss == 'mean_absolute_error':
			self.__lossfun = tf.keras.losses.mean_absolute_error
		elif loss == 'categorical_crossentropy':
			self.__lossfun = tf.keras.losses.categorical_crossentropy
		elif loss == 'binary_crossentropy':
			self.__lossfun = tf.keras.losses.binary_crossentropy
		else:
			raise ValueError(
				'an unknown loss function \'%s\' was given. ' % loss
			)

		self.__is_finished_init_train = tf.get_variable(
			'is_finished_init_train',
			shape=[],
			dtype=bool,
			initializer=tf.constant_initializer(False),
		)
		self.__x = tf.placeholder(tf.float32, shape=(None, self.__n_input_nodes), name='x')
		self.__t = tf.placeholder(tf.float32, shape=(None, self.__n_output_nodes), name='t')
		self.__alpha = tf.get_variable(
			'alpha',
			shape=[self.__n_input_nodes, self.__n_hidden_nodes],
			initializer=tf.random_uniform_initializer(-1,1),
			trainable=False,
		)
		self.__bias = tf.get_variable(
			'bias',
			shape=[self.__n_hidden_nodes],
			initializer=tf.random_uniform_initializer(-1,1),
			trainable=False,
		)
		self.__bf = tf.get_variable(
			'bias-f',
			shape=(),
			initializer=tf.random_uniform_initializer(-1,1),
			trainable=False,
		)
		self.__beta = tf.get_variable(
			'beta',
			shape=[self.__n_hidden_nodes, self.__n_output_nodes],
			initializer=tf.zeros_initializer(),
			trainable=False,
		)
		self.__bias2 = tf.get_variable(
			'bias2',
			shape=[self.__n_output_nodes],
			initializer=tf.random_uniform_initializer(-1,1),
			trainable=False,
		)
		self.__bn = tf.get_variable(
			'bias-n',
			shape=(),
			initializer=tf.random_uniform_initializer(-1,1),
			trainable=False,
		)
		self.__p = tf.get_variable(
			'p',
			shape=[self.__n_hidden_nodes, self.__n_hidden_nodes],
			initializer=tf.zeros_initializer(),
			trainable=False,
		)
		"""the previous K value"""
		self.__k = tf.get_variable(
			'k',
			shape=[self.__n_hidden_nodes, self.__n_hidden_nodes],
			initializer=tf.zeros_initializer(),
			trainable=False,
		)

		# Finish initial training phase
		self.__finish_init_train = tf.assign(self.__is_finished_init_train, True)

		# Predict
		#self.__predict = tf.matmul(self.__activation(tf.matmul(self.__x, self.__alpha) + self.__bias), self.__beta)
		self.__predict = tf.matmul(self.__activation(tf.matmul(self.__x, self.__alpha)), self.__beta)

		# Loss
		self.__loss = self.__lossfun(self.__t, self.__predict)

		# Accuracy
		self.__accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.__predict, 1), tf.argmax(self.__t, 1)), tf.float32))

		# Initial training phase
		self.__init_train = self.__build_init_train_graph()

		# Sequential training phase
		self.__seq_train = self.__build_seq_train_graph()

		# Peizhi
		# Copy Beta to Alpha phase
		self.__beta2alpha = self.__build_beta2alpha_graph()

		# Peizhi
		# Encoding
		self.__encoding = self.__activation(tf.matmul(self.__x, self.__alpha))  #  [n x m] x [m x D] = [n x D]

		# Saver
		self.__saver = tf.train.Saver()

		# Initialize variables
		self.__sess.run(tf.global_variables_initializer())

	def predict(self, x):
		return self.__sess.run(self.__predict, feed_dict={self.__x: x})

	"""get the encoding from the hidden layer"""
	def encoding(self, x):
		return self.__sess.run(self.__encoding, feed_dict={self.__x: x})


	def retrieve_alpha(self):
		"""
		Added by: Peizhi Yan
		Function: to get the encoding layer weights (alpha) 
		"""
		return self.__sess.run(self.__alpha)

	def retrieve_beta(self):
		"""
		Added by: Peizhi Yan
		Function: to get the decoding layer weights (beta) 
		"""
		return self.__sess.run(self.__beta)

	def __build_beta2alpha_graph(self):
		#'''
		#MATTHEW_FACTOR = tf.constant(0.9) # used o be 0.2 for MNIST dataset
		#MATTHEW_BETA = tf.math.scalar_mul(MATTHEW_FACTOR, tf.transpose(self.__beta))
		#MATTHEW_ALPHA = tf.math.scalar_mul(1-MATTHEW_FACTOR, self.__alpha)
		#MATTHEW_NEW_ALPHA = MATTHEW_BETA + MATTHEW_ALPHA

		#RAND = tf.random.normal(shape=[self.__n_input_nodes, self.__n_hidden_nodes],mean=0.0,stddev=0.001,dtype=tf.float32)
		#MATTHEW_NEW_ALPHA = MATTHEW_NEW_ALPHA + RAND
		#return self.__alpha.assign(MATTHEW_NEW_ALPHA)

		#self.__sess.run(self.__alpha.assign(MATTHEW_NEW_ALPHA)) # COPY weights from decoding layer to encoding layer
		#'''
		#self.__sess.run(self.__alpha.assign(tf.transpose(self.__beta))) # COPY weights from decoding layer to encoding layer
		return self.__alpha.assign(tf.transpose(self.__beta))



	def finalize(self):
		# Peizhi
		'''finalize the computational graph'''
		self.__sess.graph.finalize()

	def evaluate(self, x, t, metrics=['loss']):
		met = []
		for m in metrics:
			if m == 'loss':
				met.append(self.__loss)
			elif m == 'accuracy':
				met.append(self.__accuracy)
			else:
				return ValueError(
					'an unknown metric \'%s\' was given.' % m
				)
		ret = self.__sess.run(met, feed_dict={self.__x: x, self.__t: t})
		return list(map(lambda x: float(x), ret))

	def init_train(self, x, t):
		if self.__sess.run(self.__is_finished_init_train):
			raise Exception(
				'the initial training phase has already finished. '
				'please call \'seq_train\' method for further training.'
			)
		#i#f len(x) < self.__n_hidden_nodes:
			#raise ValueError(
				#'in the initial training phase, the number of training samples '
				#'#must be greater than the number of hidden nodes. '
				#'#But this time len(x) = %d, while n_hidden_nodes = %d' % (len(x), self.__n_hidden_nodes)
			#)
		self.__sess.run(self.__init_train, feed_dict={self.__x: x, self.__t: t})
		self.__sess.run(self.__finish_init_train) 

		#self.beta2alpha() # update: Af = An
		self.__sess.run(self.__beta2alpha)
        
	def init_train_multiple_times(self, x, t):
		if self.__sess.run(self.__is_finished_init_train):
			raise Exception(
				'the initial training phase has already finished. '
				'please call \'seq_train\' method for further training.'
			)
		if len(x) < self.__n_hidden_nodes:
			raise ValueError(
				'in the initial training phase, the number of training samples '
				'must be greater than the number of hidden nodes. '
				'But this time len(x) = %d, while n_hidden_nodes = %d' % (len(x), self.__n_hidden_nodes)
			)
		self.__sess.run(self.__init_train, feed_dict={self.__x: x, self.__t: t})
		#self.beta2alpha() # update: Af = An
		self.__sess.run(self.__beta2alpha)

	def seq_train(self, x, t):
		if self.__sess.run(self.__is_finished_init_train) == False:
			raise Exception(
				'you have not gone through the initial training phase yet. '
				'please first initialize the model\'s weights by \'init_train\' '
				'method before calling \'seq_train\' method.'
			)

		# Added by Peizhi 
		#self.__alpha.assign(tf.transpose(self.__beta)) # COPY weights from decoding layer to encoding layer
		#self.beta2alpha()
		self.__sess.run(self.__beta2alpha)

		self.__sess.run(self.__seq_train, feed_dict={self.__x: x, self.__t: t})
		

	def __build_init_train_graph(self):
		'''
		x (n samples): [n x m]
		t (n lables): [n x m]
		'''
		H = self.__activation(tf.matmul(self.__x, self.__alpha))  #  [n x m] x [m x D] = [n x D]

		'''get H's Moore-Penrose inverse'''
		HT = tf.transpose(H)  # [D x n]
		#batch_size = tf.shape(self.__x)[0]

		
		'''3*I'''
		#I = tf.eye(batch_size)  # [n x n]
		I = tf.eye(self.__n_hidden_nodes) # [D x D]
		C_I = tf.scalar_mul(self.C, I)

		
		#HHT = tf.matmul(H, HT)  # [n x n]
		HTH = tf.matmul(HT, H)  # [D x D]

		K = tf.assign(self.__k, tf.add(C_I, HTH))  # [D x D]

		#H_inverse = tf.matmul(HT, tf.matrix_inverse(K)) # H_inverse = HT(C/I+HHT)^-1  (Equation: 9)     [m x n] x [n x n] = [m x n]
		H_inverse = tf.matmul(tf.matrix_inverse(K), HT)  # [D x n] 

		'''compute beta'''
		arcsin_y = tf.math.asin(self.__t)  # [n x m]
		#inverse_acti_y =  -tf.log((tf.math.divide(tf.constant(1.0), self.__t) - tf.constant(1.0)))
		inverse_acti_y = arcsin_y

		An = tf.assign(self.__beta, tf.matmul(H_inverse, inverse_acti_y))  # [D x n] x [n x m] = [D x m]

		'''compute beta bias'''
		AnHf = tf.matmul(H, An)  # [n x D] x [D x m] = [n x m]
		Bn = tf.losses.mean_squared_error(AnHf, inverse_acti_y)  # [1 x m]
		Bn = tf.sqrt(Bn)  # [1 x m]

		'''update Bf'''
		init_train = tf.assign(self.__bf, Bn)

		'''Af is updated in init_train, not here!!'''

		#init_train = tf.assign(self.__beta, pHTt)  # beta <-- pHTt
		return init_train # Beta_0

	def __build_seq_train_graph(self):
		'''
		x (n samples): [n x m]
		t (n lables): [n x m]
		'''

		'''H(k+1)'''
		H = self.__activation(tf.matmul(self.__x, self.__alpha))  #  [n x m] x [m x D] = [n x D]

		'''get H(t+1) transpose'''
		HT = tf.transpose(H)  # [D x n]

		#HHT = tf.matmul(H, HT)  # [n x n]
		HTH = tf.matmul(HT, H)  # [D x D]


		'''K(k+1)'''
		K = tf.assign(self.__k,  self.__k + HTH)  # [D x D]
		
		'''K^-1'''
		K_inverse = tf.matrix_inverse(K)  # [D x D]


		"""Inverse of activation function"""
		arcsin_y = tf.math.asin(self.__t)  # [n x m]      if activation function is sin()
		#inverse_acti_y = -tf.log((tf.math.divide(tf.constant(1.0), self.__t) - tf.constant(1.0)))
		inverse_acti_y = arcsin_y


		#     ([D x D]*[D x n])*([n x m] - ([n x D]*[D x m]))
		# ==>   [D x n]*[n x m]
		# ==>   [D x m]
		UPDATE = tf.matmul(tf.matmul(K_inverse, HT), inverse_acti_y - tf.matmul(H, self.__beta))  # [D x m]



		###############
		"""TESTING!!"""
		#LR = tf.constant(1.1)
		#UPDATE = tf.math.scalar_mul(LR, UPDATE)



		An = tf.assign(self.__beta, self.__beta + UPDATE)  # [D x m]


		'''compute beta bias'''
		AnHf = tf.matmul(H, An)  # [n x D] x [D x m] = [n x m]
		Bn = tf.losses.mean_squared_error(AnHf, inverse_acti_y)  # [1 x m]
		Bn = tf.sqrt(Bn)  # [1 x m]
		Bn = tf.assign(self.__bn, Bn)
		
		'''update Bf'''
		seq_train = tf.assign(self.__bf, Bn)
				
		return seq_train # Beta_k

	def save(self, filepath):
		self.__saver.save(self.__sess, filepath)

	def restore(self, filepath):
		self.__saver.restore(self.__sess, filepath)

	def initialize_variables(self):
		for var in [self.__alpha, self.__bias, self.__beta, self.__p, self.__is_finished_init_train]:
			self.__sess.run(var.initializer)

	def __del__(self):
		self.__sess.close()

	@property
	def input_shape(self):
		return (self.__n_input_nodes,)

	@property
	def output_shape(self):
		return (self.__n_output_nodes,)

	@property
	def n_input_nodes(self):
		return self.__n_input_nodes

	@property
	def n_hidden_nodes(self):
		return self.__n_hidden_nodes

	@property
	def n_output_nodes(self):
		return self.__n_output_nodes
