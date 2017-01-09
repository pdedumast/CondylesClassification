import os
import sys
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
import pandas as pd


nbPoints = 1002
nbLabels = 6
nbFeatures = 3 + nbLabels + 4

if nbLabels == 6:
	saveModel = 'weights_5Groups.ckpt'
else:
	saveModel = 'weights_7Groups.ckpt'

# ----------------------------------------------------------------------------- #
#                                 Needed Functions
# ----------------------------------------------------------------------------- #
## Reformat into a shape that's more adapted to the models we're going to train:
#   - data as a flat matrix
#   - labels as float 1-hot encodings
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, nbPoints * nbFeatures)).astype(np.float32)
	labels = (np.arange(nbLabels) == labels[:,None]).astype(np.float32)
	return dataset, labels


## Performance measures of the network
# Computation of : 	- Accuracy
# 					- Precision (PPV)
# 					- Sensitivity (TPR)
# 					- Confusion matrix
def accuracy(predictions, labels):
	# Accuracy
	accuracy = (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

	# Confusion matrix
	[nbSamples, column] = predictions.shape
	actu = np.zeros(nbSamples)
	pred = np.zeros(nbSamples)
	for i in range(0, nbSamples):
		actu[i] = np.argmax(labels[i,:])
		pred[i] = np.argmax(predictions[i,:])
	y_actu = pd.Series(actu, name='Actual')
	y_pred = pd.Series(pred, name='Predicted')
	df_confusion = pd.crosstab(y_actu, y_pred)

	# PPV and TPR
	TruePos_sum = int(np.sum(predictions[:, 1] * labels[:, 1]))
	PredPos_sum = int(max(np.sum(predictions[:, 1]), 1)) #Max to avoid to divide by 0
	PredNeg_sum = np.sum(predictions[:, 0])
	RealPos_sum = int(np.sum(labels[:, 1]))

	if not PredPos_sum :
		PPV = 0
	else:
		PPV = 100.0 *TruePos_sum / PredPos_sum # Positive Predictive Value, Precision
	if not RealPos_sum:
		TPR = 0
	else:
		TPR = 100.0 *TruePos_sum / RealPos_sum  # True Positive Rate, Sensitivity

	return accuracy, df_confusion, PPV, TPR


# ----------------------------------------------------------------------------- #
# 									PROGRAM										#
# ----------------------------------------------------------------------------- #

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


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print("\nTraining set", train_dataset.shape, train_labels.shape)
print("Validation set", valid_dataset.shape, valid_labels.shape)
print("Test set", test_dataset.shape, test_labels.shape)

## Network and training parameters
learning_rate = 0.0005
batch_size = 10
nb_hidden_layers = 3

if nb_hidden_layers == 2:
	nb_hidden_nodes_1 = 2048
elif nb_hidden_layers == 3:
	nb_hidden_nodes_1, nb_hidden_nodes_2 = 2048, 2048

regularization = True
lambda_reg = 0.01

num_steps = 1001
num_epochs = 2

print "\nnbGroups = " + str(nbLabels)
print "nb_hidden_layers = " + str(nb_hidden_layers)
print "regularization : " + str(regularization)
print "num_steps = " + str(num_steps)
print "num_epochs = " + str(num_epochs)
print "batch_size = " + str(batch_size) + "\n"


# ----------------------------------------------------------------------------- #
# 																				#
# 						   Passons aux choses serieuses							#
# 																				#
# ----------------------------------------------------------------------------- #
# 

graph = tf.Graph()
with graph.as_default():
	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, nbPoints * nbFeatures))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, nbLabels))

	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	def weight_variable(shape, name=None):
		"""Create a weight variable with appropriate initialization."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, name = name)

	def bias_variable(shape, name=None):
		"""Create a bias variable with appropriate initialization."""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial,name = name)

	if nb_hidden_layers == 1:
		W_fc1 = weight_variable([nbPoints * nbFeatures, nbLabels], "W_fc1")
		b_fc1 = bias_variable([nbLabels],"b_fc1")

	elif nb_hidden_layers == 2:
		W_fc1 = weight_variable([nbPoints * nbFeatures, nb_hidden_nodes_1], "W_fc1")
		b_fc1 = bias_variable([nb_hidden_nodes_1],"b_fc1")

		W_fc2 = weight_variable([nb_hidden_nodes_1, nbLabels], "W_fc2")
		b_fc2 = bias_variable([nbLabels],"b_fc2")

	elif nb_hidden_layers == 3:
		W_fc1 = weight_variable([nbPoints * nbFeatures, nb_hidden_nodes_1], "W_fc1")
		b_fc1 = bias_variable([nb_hidden_nodes_1],"b_fc1")

		W_fc2 = weight_variable([nb_hidden_nodes_1, nb_hidden_nodes_2], "W_fc2")
		b_fc2 = bias_variable([nb_hidden_nodes_2],"b_fc2")

		W_fc3 = weight_variable([nb_hidden_nodes_2, nbLabels], "W_fc3")
		b_fc3 = bias_variable([nbLabels],"b_fc3")

	# Model.
	def model(data):

		if nb_hidden_layers == 1:
			with tf.name_scope('FullyConnected1'):
				h_fc1 = tf.matmul(data, W_fc1) + b_fc1
				return h_fc1

		elif nb_hidden_layers == 2:
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

	# Training computation.
	logits = model(tf_train_dataset)
	
	if not regularization:
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	else:
		if nb_hidden_layers == 1:
			norms = tf.nn.l2_loss(W_fc1) 
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + lambda_reg*norms)
		elif nb_hidden_layers == 2:
			norms = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + lambda_reg*norms)
		elif nb_hidden_layers == 3:
			norms = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + lambda_reg*norms)

	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	# optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

	if nb_hidden_layers == 2:
		saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2})
	elif nb_hidden_layers == 3:
	 	saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2, "W_fc3": W_fc3, "b_fc3": b_fc3})
		
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))


	




	# -------------------------- #
	#		Let's run it 		 #
	# -------------------------- #
	# 
	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()
		print("Initialized")
		for epoch in range(0, num_epochs):
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

		finalaccuracy, mat_confusion, PPV, TPR = accuracy(test_prediction.eval(), test_labels)
		print("Test accuracy: %.1f%%" % finalaccuracy)
		print("\n\nConfusion matrix :\n" + str(mat_confusion))
		print "\n PPV : " + str(PPV)
		print "\n TPR : " + str(TPR)

		if saveModel.rfind(".ckpt") != -1:
			save_path = saver.save(session, saveModel)
			print("Model saved in file: %s" % save_path)
		else:
			raise Exception("Impossible to save train model at %s. Must be a .cpkt file" % saveModelPath)

		summary_writer = tf.train.SummaryWriter('.', graph=session.graph)










