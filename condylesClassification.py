import os
import sys
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
# import vtk
import pandas as pd


saveModelPath = 'saved_weights.ckpt'


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

nbPoints = 1002
nbFeatures = 15
nbLabels = 6

# featuresType = "norm"             # "norm" : que les normales, 3 composantes
# featuresType = "norm-pos"         # "norm-pos" : normales + positions, 6 composantes
# featuresType = "norm-dist"        # "norm-dist" : normales + distances aux mean. (3+nbGroups) composantes
featuresType = "norm-dist-curv"     # "norm-dist-curv" : normales + distances aux mean + curvatures. (3+nbGroups+4) composantes
# featuresType = "norm-curv"        # "norm-curv" : normales + curvatures. (3+4) composantes

if featuresType == "norm":
    nbFeatures = 3
elif featuresType == "norm-pos":
    nbFeatures = 3 + 3
elif featuresType == "norm-dist":
    nbFeatures = 3 + nbLabels
elif featuresType == "norm-dist-curv":
    nbFeatures = 3 + nbLabels + 4
elif featuresType == "norm-curv":
    nbFeatures = 3 + 4 


def reformat(dataset, labels):
	dataset = dataset.reshape((-1, nbPoints * nbFeatures)).astype(np.float32)
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

	[nbSamples, column] = predictions.shape
	actu = np.zeros(nbSamples)
	pred = np.zeros(nbSamples)

	for i in range(0, nbSamples):
		actu[i] = np.argmax(predictions[i,:])
		pred[i] = np.argmax(labels[i,:])

	# print "actu : " + str(actu)
	# print "pred : " + str(pred)

	y_actu = pd.Series(actu, name='Actual')
	y_pred = pd.Series(pred, name='Predicted')
	df_confusion = pd.crosstab(y_actu, y_pred)
	
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0]), df_confusion


# ----------------------------------------------------------------------------- #
# 																				#
# 						   Passons aux choses serieuses							#
# 																				#
# ----------------------------------------------------------------------------- #
# 

learning_rate = 0.0005
batch_size = 10
nb_hidden_layers = 3

if nb_hidden_layers == 2:
	nb_hidden_nodes_1 = 10000
elif nb_hidden_layers == 3:
	nb_hidden_nodes_1 = 2048
	nb_hidden_nodes_2 = 2048
elif nb_hidden_layers == 5:
	nb_hidden_nodes_1 = 2048
	nb_hidden_nodes_2 = 1024
	nb_hidden_nodes_3 = 1024
	nb_hidden_nodes_4 = 512

regularization = True
# regularization = False
lambda_reg = 0.01

graph = tf.Graph()
with graph.as_default():
	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, nbPoints * nbFeatures))
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

	if nb_hidden_layers == 2:
		W_fc1 = weight_variable([nbPoints * nbFeatures, nb_hidden_nodes_1])
		b_fc1 = bias_variable([nb_hidden_nodes_1])

		W_fc2 = weight_variable([nb_hidden_nodes_1, nbLabels])
		b_fc2 = bias_variable([nbLabels])

	elif nb_hidden_layers == 3:
		W_fc1 = weight_variable([nbPoints * nbFeatures, nb_hidden_nodes_1])
		b_fc1 = bias_variable([nb_hidden_nodes_1])

		W_fc2 = weight_variable([nb_hidden_nodes_1, nb_hidden_nodes_2])
		b_fc2 = bias_variable([nb_hidden_nodes_2])

		W_fc3 = weight_variable([nb_hidden_nodes_2, nbLabels])
		b_fc3 = bias_variable([nbLabels])

	elif nb_hidden_layers == 5:
		W_fc1 = weight_variable([nbPoints * nbFeatures, nb_hidden_nodes_1])
		b_fc1 = bias_variable([nb_hidden_nodes_1])

		W_fc2 = weight_variable([nb_hidden_nodes_1, nb_hidden_nodes_2])
		b_fc2 = bias_variable([nb_hidden_nodes_2])

		W_fc3 = weight_variable([nb_hidden_nodes_2, nb_hidden_nodes_3])
		b_fc3 = bias_variable([nb_hidden_nodes_3])

		W_fc4 = weight_variable([nb_hidden_nodes_3, nb_hidden_nodes_4])
		b_fc4 = bias_variable([nb_hidden_nodes_4])

		W_fc5 = weight_variable([nb_hidden_nodes_4, nbLabels])
		b_fc5 = bias_variable([nbLabels])

	# Model.
	def model(data):

		if nb_hidden_layers == 2:
			with tf.name_scope('FullyConnected1'):

				h_fc1 = tf.matmul(data, W_fc1) + b_fc1
				h_relu1 = tf.nn.relu(h_fc1)

			with tf.name_scope('FullyConnected2'):

				h_fc2 = tf.matmul(h_relu1, W_fc2) + b_fc2
				return h_fc2

		elif nb_hidden_layers == 3:
			with tf.name_scope('FullyConnected1'):

				h_fc1 = tf.matmul(data, W_fc1) + b_fc1
				h_relu1 = tf.nn.relu(h_fc1)

			with tf.name_scope('FullyConnected1'):

				h_fc2 = tf.matmul(h_relu1, W_fc2) + b_fc2
				h_relu2 = tf.nn.relu(h_fc2)

			with tf.name_scope('FullyConnected3'):

				h_fc3 = tf.matmul(h_relu2, W_fc3) + b_fc3
				return h_fc3


		elif nb_hidden_layers == 5:
			with tf.name_scope('FullyConnected1'):

				h_fc1 = tf.matmul(data, W_fc1) + b_fc1
				h_relu1 = tf.nn.relu(h_fc1)

				# print "\nInput dimension FC 1: " + str(data.get_shape())
				# print "Output dimension FC 1: " + str(h_relu1.get_shape())

			with tf.name_scope('FullyConnected2'):

				h_fc2 = tf.matmul(h_relu1, W_fc2) + b_fc2
				h_relu2 = tf.nn.relu(h_fc2)

				# print "\nInput dimension FC 2: " + str(h_relu1.get_shape())
				# print "Output dimension FC 2: " + str(h_relu2.get_shape())

			with tf.name_scope('FullyConnected3'):

				h_fc3 = tf.matmul(h_relu2, W_fc3) + b_fc3
				h_relu3 = tf.nn.relu(h_fc3)

				# print "\nInput dimension FC 3: " + str(h_relu2.get_shape())
				# print "Output dimension FC 3: " + str(h_relu3.get_shape())

			with tf.name_scope('FullyConnected4'):

				h_fc4 = tf.matmul(h_relu3, W_fc4) + b_fc4
				h_relu4 = tf.nn.relu(h_fc4)

				# print "\nInput dimension FC 4: " + str(h_relu3.get_shape())
				# print "Output dimension FC 4: " + str(h_relu4.get_shape())

			with tf.name_scope('FullyConnected5'):

				h_fc5 = tf.matmul(h_relu4, W_fc5) + b_fc5

				# print "\nInput dimension FC 5: " + str(h_relu4.get_shape())
				# print "Output dimension FC 5: " + str(h_fc5.get_shape())

			return h_fc5

	# Training computation.
	logits = model(tf_train_dataset)
	
	if not regularization:
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	else:
		if nb_hidden_layers == 2:
			norms = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + lambda_reg*norms)
		elif nb_hidden_layers == 3:
			norms = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + lambda_reg*norms)
		elif nb_hidden_layers == 5:
			# norms = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(b_fc2) + tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(b_fc3) + tf.nn.l2_loss(W_fc4) + tf.nn.l2_loss(b_fc4) + tf.nn.l2_loss(W_fc5) + tf.nn.l2_loss(b_fc5)
			norms = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(W_fc4) + tf.nn.l2_loss(W_fc5)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + lambda_reg*norms)

	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	# optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
	
	if nb_hidden_layers == 2:
		saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2})
	elif nb_hidden_layers == 3:
		saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2, "W_fc3": W_fc3, "b_fc3": b_fc3})
	elif nb_hidden_layers == 5:
		saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2, "W_fc3": W_fc3, "b_fc3": b_fc3, "W_fc4": W_fc4, "b_fc4": b_fc4, "W_fc5": W_fc5, "b_fc5": b_fc5})

	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))


##### Let's run it: #####
# num_steps = 2001
# num_steps = 1001
num_steps = 501

print ""
print "nbGroups = " + str(nbLabels)
print "nb_hidden_layers = " + str(nb_hidden_layers)
print "regularization : " + str(regularization)
print "num steps = " + str(num_steps)
print ""

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
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)[0])
			print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels)[0])
	finalaccuracy = accuracy(test_prediction.eval(), test_labels)
	print("Test accuracy: %.1f%%" % finalaccuracy[0])
	print("\n\nConfusion matrix :")
	print finalaccuracy[1]
	print ""

	# Save the variables to disk.
	tf.Print(W_fc1)
	if saveModelPath.rfind(".ckpt") != -1:
		save_path = saver.save(session, saveModelPath)
		print("Model saved in file: %s" % save_path)
	else:
		raise Exception("Impossible to save train model at %s. Must be a .cpkt file" % saveModelPath)















