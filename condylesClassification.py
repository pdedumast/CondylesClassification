import os
import sys
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
# import vtk



# --------------------------------------------------------------------------------------------------- #
# Reoad the data generated in pickleData.py

pickle_file = 'condyles.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
  	train_dataset = save['train_dataset']
  	train_labels = save['train_labels']
  	valid_dataset = save['valid_dataset']
  	valid_labels = save['valid_labels']
  	test_dataset = save['test_dataset']
  	test_labels = save['test_labels']
  	del save  # hint to help gc free up memory
  	print('Training set', train_dataset.shape, train_labels.shape)
  	print('Validation set', valid_dataset.shape, valid_labels.shape)
  	print('Test set', test_dataset.shape, test_labels.shape)


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
## Reformat into a shape that's more adapted to the models we're going to train:
#   - data as a flat matrix
#   - labels as float 1-hot encodings

nbPoints = 1000  
nbFeatures = 6
nbLabels = 2
nbChannels = 1	# Input depth


def reformat(dataset, labels):
	dataset = dataset.reshape((-1, nbPoints, nbFeatures, nbChannels)).astype(np.float32)
	labels = (np.arange(nbLabels) == labels[:,None]).astype(np.float32)
	return dataset, labels

# ----------------------------------------------------------------------------- #

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('\nTraining set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# ----------------------------------------------------------------------------- #

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])



# ----------------------------------------------------------------------------- #
# 																				#
# 						   Passons aux choses serieuses							#
# 																				#
# ----------------------------------------------------------------------------- #
# 
# SGD
# 1-hidden layer
# 1024 hidden nodes
# 
learning_rate = 0.005
batch_size = 15
nb_hidden_layers = 1024
patch_size1  = 5
patch_size2  = 3
nbFilterLayer = 16


graph = tf.Graph()
with graph.as_default():
	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, nbPoints, nbFeatures, nbChannels))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, nbLabels))

	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	def weight_variable(shape):
		"""Create a weight variable with appropriate initialization."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		"""Create a bias variable with appropriate initialization."""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	# Variables.
	W_conv1 = weight_variable([patch_size1, patch_size1, nbChannels, nbFilterLayer])
	b_conv1 = bias_variable([nbFilterLayer])

	W_conv2 = weight_variable([patch_size2, patch_size2, nbFilterLayer, nbFilterLayer])
	b_conv2 = bias_variable([nbFilterLayer])	

	# W_fc1h = weight_variable([nbPoints * nbFeatures * nbFilterLayer, nb_hidden_layers])
	# b_fc1h = bias_variable([nb_hidden_layers])

	# W_fc1 = weight_variable([nb_hidden_layers, nbLabels])
	# b_fc1 = bias_variable([nbLabels])
	
	W_fc1 = weight_variable([nbPoints // 4 * nbFeatures // 2 * nbFilterLayer, nbLabels])
	b_fc1 = bias_variable([nbLabels])

	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def max_pool_2x1(x):
		return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')


	# Model.
	def model(data):
		with tf.name_scope('Conv1'):
			h_conv1 = conv2d(data, W_conv1)
			h_relu1 = tf.nn.relu(h_conv1 + b_conv1)

			print "\nInput dimension Conv 1: " + str(data.get_shape())
			print "Output dimension Conv 1: " + str(h_relu1.get_shape())

		with tf.name_scope('MaxPooling1'):
			h_maxpool1 = max_pool_2x2(h_relu1)

			print "\nInput dimension Max-Pooling 1: " + str(h_relu1.get_shape())
			print "Output dimension Max-Pooling 1: " + str(h_maxpool1.get_shape())

		with tf.name_scope('Conv2'):
			h_conv2 = conv2d(h_maxpool1, W_conv2)
			h_relu2 = tf.nn.relu(h_conv2 + b_conv2)

			print "\nInput dimension Conv 2: " + str(h_maxpool1.get_shape())
			print "Output dimension Conv 2: " + str(h_conv2.get_shape())

		with tf.name_scope('MaxPooling2'):
			h_maxpool2 = max_pool_2x1(h_relu2)

			print "\nInput dimension Max-Pooling 2: " + str(h_relu2.get_shape())
			print "Output dimension Max-Pooling 2: " + str(h_maxpool2.get_shape())


		with tf.name_scope('FullyConnected1'):

			shape = h_maxpool2.get_shape().as_list()
			reshape = tf.reshape(h_maxpool2, [shape[0], shape[1] * shape[2] * shape[3]])

			# h_relu2 = tf.nn.relu(tf.matmul(reshape, W_fc1h) + B_fc1h)
			# output = tf.matmul(h_relu2, W_fc1) + B_fc1
			# print ""
			# print "Input dimension FC1: " + str(h_relu1.get_shape())
			# print "Hidden dimension FC1: " + str(h_relu2.get_shape())
			# print "Output dimension FC1: " + str(output.get_shape())

			output = tf.matmul(reshape, W_fc1) + b_fc1

			print "\nInput dimension FC 1: " + str(h_maxpool2.get_shape())
			print "Output dimension FC 1: " + str(output.get_shape())

		return output

	# Training computation.
	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

	# Optimizer.
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))




##### Let's run it: #####
num_steps = 3001

with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	print("Initialized")
	for step in range(num_steps):
		# Pick an offset within the training data, which has been randomized.
		# Note: we could use better randomization across epochs.
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		# Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		# Prepare a dictionary telling the session where to feed the minibatch.
		# The key of the dictionary is the placeholder node of the graph to be fed,
		# and the value is the numpy array to feed to it.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 500 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))














