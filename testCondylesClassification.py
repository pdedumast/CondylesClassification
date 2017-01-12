import os
import sys
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
import pandas as pd

import neuralnetwork as nn



# ----------------------------------------------------------------------------- #
# 									PROGRAM										#
# ----------------------------------------------------------------------------- #


saveModelPath = 'weights_5Groups.ckpt'


if nn.NUM_HIDDEN_LAYERS == 1:
	nb_hidden_nodes_1 = 2048
elif nn.NUM_HIDDEN_LAYERS == 2:
	nb_hidden_nodes_1, nb_hidden_nodes_2 = 2048, 2048

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_float('lambda_reg', 0.01, 'Regularization lambda factor.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_steps', 1001, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_string('pickle_file', 'condyles.pickle','Path to the pickle file containing datasets')

# ---------------------------------------------------------------------------- #
# Reoad the data generated in pickleData.py


with open(FLAGS.pickle_file, 'rb') as f:
	save = pickle.load(f)
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save  # hint to help gc free up memory
	print('Test set', test_dataset.shape, test_labels.shape)

nbData = len(test_dataset[:, 0, 0])
test_dataset = test_dataset.reshape((-1, nn.NUM_FEATURES * nn.NUM_POINTS)).astype(np.float32)
predictions = np.ndarray(shape=(nbData, nn.NUM_CLASSES), dtype=np.float32)

test_dataset, test_labels = nn.reformat(test_dataset, test_labels)

print('Test set', test_dataset.shape, test_labels.shape)


# ----------------------------------------------------------------------------- #
# 																				#
# 						   Passons aux choses serieuses							#
# 																				#
# ----------------------------------------------------------------------------- #
# 
def run_test():

	# Construct the graph
	graph = tf.Graph()
	with graph.as_default():
		# Input data.
		with tf.name_scope('Inputs_management'):
			tf_test_dataset = tf.constant(test_dataset, name="tf_test_dataset")

		with tf.name_scope('Bias_and_weights_management'):
			weightsDict = nn.bias_weights_creation(nb_hidden_nodes_1, nb_hidden_nodes_2)	
		# Training computation.
		with tf.name_scope('Network_computations'):
			logits,weightsDict = nn.model(tf_test_dataset, weightsDict)
			
		with tf.name_scope('Predictions'):
			# Predictions for the training, validation, and test data.
			test_prediction = tf.nn.softmax(logits)

		with tf.name_scope('Restore_weights'):
			saver = tf.train.Saver(weightsDict)

	with tf.Session(graph=graph) as session:

		# create log writer object
		writer = tf.train.SummaryWriter('./test', graph=graph)

		saver.restore(session, saveModelPath)
		feed_dict = {tf_test_dataset: test_dataset}
		predictions = session.run(test_prediction, feed_dict=feed_dict)

		finalaccuracy, mat_confusion, PPV, TPR = nn.accuracy(predictions, test_labels)
		
		print("Test accuracy: %.1f%%" % finalaccuracy)
		print("\n\nConfusion matrix :\n" + str(mat_confusion))
		print "\n PPV : " + str(PPV)
		print "\n TPR : " + str(TPR)



def main(_):
	run_test()

if __name__ == '__main__':
    tf.app.run()






