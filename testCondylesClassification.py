import os
import sys
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
# import vtk
import pandas as pd


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

saveModelPath = 'saved_weights.ckpt'

nbPoints = 1002
nbLabels = 6
nb_hidden_layers = 2
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


if nb_hidden_layers == 2:
	nb_hidden_nodes_1 = 2048
elif nb_hidden_layers == 3:
	nb_hidden_nodes_1 = 2048
	nb_hidden_nodes_2 = 2048

# ---------------------------------------------------------------------------- #
# Reoad the data generated in pickleData.py
pickle_file = 'condyles.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save  # hint to help gc free up memory
	print('Test set', test_dataset.shape, test_labels.shape)

nbData = len(test_dataset[:, 0, 0])
test_dataset = test_dataset.reshape((-1, nbFeatures * nbPoints)).astype(np.float32)
predictions = np.ndarray(shape=(nbData, nbLabels), dtype=np.float32)

test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Test set', test_dataset.shape, test_labels.shape)


# ----------------------------------------------------------------------------- #
# 																				#
# 						   Passons aux choses serieuses							#
# 																				#
# ----------------------------------------------------------------------------- #
# 


graph = tf.Graph()
with graph.as_default():
    # Input data.
	tf_test_dataset = tf.constant(test_dataset)

	def weight_variable(shape, name=None):
		"""Create a weight variable with appropriate initialization."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, name=name)

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

	# Training computation. Pas de regularization pour le test
	test_prediction = tf.nn.softmax(model(tf_test_dataset))
    
    # Add ops to restore variables.
	if nb_hidden_layers == 2:
		saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2})
	elif nb_hidden_layers == 3:
	 	saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2, "W_fc3": W_fc3, "b_fc3": b_fc3})
		

	with tf.Session(graph=graph) as session:
		saver.restore(session, saveModelPath)
		feed_dict = {tf_test_dataset: test_dataset}
		finalaccuracy, mat_confusion, PPV, TPR = accuracy(test_prediction.eval(), test_labels)
		
		print("Test accuracy: %.1f%%" % finalaccuracy)
		print("\n\nConfusion matrix :\n" + str(mat_confusion))
		print "\n PPV : " + str(PPV)
		print "\n TPR : " + str(TPR)








