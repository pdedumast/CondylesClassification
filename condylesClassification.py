import os
import sys
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
# import vtk
import pandas as pd
from random import randint

saveModelPath = 'saved_weights.ckpt'


# ----------------------------------------------------------------------------- #

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

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #


if nbLabels == 8:
    location = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/7Groups-Feat/"
elif nbLabels == 6:
    location = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/5Groups-Feat/"

train_folders = [os.path.join(location, d) for d in sorted(os.listdir(location))]       # Folder class liste

# Delete .DS_Store file if there is one
if train_folders.count(str(location) + ".DS_Store"):
    train_folders.remove(str(location) + ".DS_Store")

# Getthe list of all .pickle files
train_datasets = list()
for f in train_folders:
	_, ext = os.path.splitext(f)
	if ext == ".pickle":
		train_datasets.append(f)

test_datasets = train_datasets
print ""




# ----------------------------------------------------------------------------- #

#
# Function make_arrays(nb_rows, nbPoints, nbFeatures)
#
#
def make_arrays(nb_rows, nbPoints, nbFeatures):
	if nb_rows:
		dataset = np.ndarray((nb_rows, nbPoints, nbFeatures), dtype=np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset, labels = None, None
	return dataset, labels


#########
# Create list of array. 1 array per group. Each group shuffled

train_datasets_list = list()
train_labels_list = list()

for label,pickle_file in enumerate(train_datasets):
	try:
		with open(pickle_file, 'rb') as f:
			shape_set = pickle.load(f)
			np.random.shuffle(shape_set)
			train_dataset, train_labels = make_arrays(shape_set.shape[0], nbPoints, nbFeatures)
			train_datasets_list.append(train_dataset)
			train_labels_list.append(train_labels)

	except Exception as e:
		print('Unable to process data from', pickle_file, ':', e)
		raise

# Create the TEST dataset - Contains ALL the data (209)
def create_test_dataset(train_datasets_list, min_test_size):
	num_classes = len(train_datasets_list)
	test_dataset, test_labels = make_arrays(min_test_size, nbPoints, nbFeatures)

	start_t, end_t = 0, 0
	for label, dataset in enumerate(train_datasets_list):
		np.random.shuffle(dataset)

		end_t += dataset.shape[0]
		test_shapes = dataset[:, :, :]
		test_dataset[start_t:end_t, :, :] = test_shapes
		test_labels[start_t:end_t] = label
		start_t += dataset.shape[0]
		
	return test_dataset, test_labels

test_dataset, test_labels = create_test_dataset(train_datasets_list, 209)		# 209 = entire dataset size


#
# Function randomize(dataset, labels)
#   Randomize the data and their labels
#
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

test_dataset, test_labels = randomize(test_dataset, test_labels)



# ---------
# Create the VALID and TRAIN dataset - TRAIN + VALID = all the dataset. 
# Data in Valid_dataset CAN'T be in Train_dataset
def create_valid_train_dataset(pickle_files, vsize_per_class):
	num_classes = len(train_datasets_list)
	valid_dataset, valid_labels = make_arrays(vsize_per_class * num_classes, nbPoints, nbFeatures)
	train_dataset, train_labels = make_arrays(209 - vsize_per_class * num_classes, nbPoints, nbFeatures)

	start_v, start_t = 0, 0
	end_v, end_t = vsize_per_class, 0

	for label, pickle_file in enumerate(pickle_files):
		try:
			with open(pickle_file, 'rb') as f:
				shape_set = pickle.load(f)
				# let's shuffle the letters to have random validation and training set
				np.random.shuffle(shape_set)

				valid_shapes = shape_set[:vsize_per_class, :, :]
				valid_dataset[start_v:end_v, :, :] = valid_shapes
				valid_labels[start_v:end_v] = label

				end_t += shape_set.shape[0] - vsize_per_class

				train_shapes = shape_set[vsize_per_class:, :, :]
				train_dataset[start_t:end_t, :, :] = train_shapes
				train_labels[start_t:end_t] = label

				start_v += vsize_per_class
				end_v += vsize_per_class
				start_t += shape_set.shape[0] - vsize_per_class

				print "Done label : " + str(label)

		except Exception as e:
			print('Unable to process data from', pickle_file, ':', e)
			raise

	return train_dataset, train_labels, valid_dataset, valid_labels

train_dataset, train_labels, valid_dataset, valid_labels = create_valid_train_dataset(train_datasets, 2)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# ----------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------- #
## Reformat into a shape that's more adapted to the models we're going to train:
#   - data as a flat matrix
#   - labels as float 1-hot encodings
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, nbPoints * nbFeatures)).astype(np.float32)
	labels = (np.arange(nbLabels) == labels[:,None]).astype(np.float32)
	return dataset, labels

# ----------------------------------------------------------------------------- #

valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# ----------------------------------------------------------------------------- #

#
# Compute Accuray, precision, Sensitivity and Confusion matrix
# 
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
# 																				#
# 						   Passons aux choses serieuses							#
# 																				#
# ----------------------------------------------------------------------------- #
# 

learning_rate = 0.0005
batch_size = 18
nb_hidden_layers = 2

if nb_hidden_layers == 2:
	nb_hidden_nodes_1 = 2048
elif nb_hidden_layers == 3:
	nb_hidden_nodes_1 = 2048
	nb_hidden_nodes_2 = 2048


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

	if nb_hidden_layers == 1:
		W_fc1 = weight_variable([nbPoints * nbFeatures, nbLabels])
		b_fc1 = bias_variable([nbLabels])

	elif nb_hidden_layers == 2:
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



# ----------------------------------------------------------------------------- #

# Create a minibatch for the training
# Same number of data from each class
# 
def create_minibatch(train_dataset, tsize_per_class):
	batch_dataset, batch_labels = make_arrays(nbLabels * tsize_per_class, nbPoints, nbFeatures)

	for i in range(0, nbLabels):
		index = np.where(train_labels == i)
	
		ind_prev, ind = -10, -10
		j = 0

		while j < tsize_per_class:
			while (ind == ind_prev):
				ind = randint(np.argmin(index), np.argmax(index))

			batch_dataset[2 * i + j, :, :] = train_dataset[ind, :, :]
			batch_labels[2 * i + j] = i
			j = j + 1

	return batch_dataset, batch_labels




# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #


##### Let's run it: #####
# num_steps = 2001
# num_steps = 1001
num_steps = 1001
num_epochs = 2


print ""
print "nbGroups = " + str(nbLabels)
print "nb_hidden_layers = " + str(nb_hidden_layers)
print "regularization : " + str(regularization)
print "num steps = " + str(num_steps)
print "num epochs = " + str(num_epochs)
print "batch_size = " + str(batch_size)
print ""

with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	print("Initialized")
	for epoch in range(0, num_epochs):
		for step in range(num_steps):

			batch_data, batch_labels = create_minibatch(train_dataset, 3)
			batch_data, batch_labels = randomize(batch_data, batch_labels)
			batch_data, batch_labels = reformat(batch_data, batch_labels)

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

	print "\nThis is W_fc1 : "
	print (session.run(W_fc1[:,:]))

	print "\nThis is W_fc2 : "
	print (session.run(W_fc2[:,:]))

	# print "\nThis is W_fc3 : "
	# print (session.run(W_fc3))

	# Save the variables to disk.
	if saveModelPath.rfind(".ckpt") != -1:
		save_path = saver.save(session, saveModelPath)
		print("Model saved in file: %s" % save_path)
	else:
		raise Exception("Impossible to save train model at %s. Must be a .cpkt file" % saveModelPath)















