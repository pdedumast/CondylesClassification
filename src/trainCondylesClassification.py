import os
import sys
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
# import pandas as pd
import neuralnetwork as nn
import inputdata


print "VERSION ::: " + str(tf.__version__)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_float('lambda_reg', 0.01, 'Regularization lambda factor.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_steps', 101, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-inputPickle', action='store', dest='pickle_file', help='Input file to classify', 
                    default = "/Users/prisgdd/Documents/Projects/CNN/CondylesClassification/condyles.pickle")

parser.add_argument('-saveModelPath', action='store', dest='saveModelPath', help='Path to the saved model to use', default='weights_5Groups')

args = parser.parse_args()
pickle_file = args.pickle_file
saveModelPath = args.saveModelPath


# ----------------------------------------------------------------------------- #
# 									PROGRAM										#
# ----------------------------------------------------------------------------- #
def get_inputs(pickle_file):

	# Reoad the data generated in pickleData.py
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
	  	# print('Test set', test_dataset.shape, test_labels.shape)

		train_dataset, train_labels = inputdata.reformat(train_dataset, train_labels)
		valid_dataset, valid_labels = inputdata.reformat(valid_dataset, valid_labels)
		# test_dataset, test_labels = inputdata.reformat(test_dataset, test_labels)
		print("\nTraining set", train_dataset.shape, train_labels.shape)
		print("Validation set", valid_dataset.shape, valid_labels.shape)
		# print("Test set", test_dataset.shape, test_labels.shape)

		return train_dataset, train_labels, valid_dataset, valid_labels

## Network and training parameters


if nn.NUM_HIDDEN_LAYERS == 1:
	nb_hidden_nodes_1 = 2048
elif nn.NUM_HIDDEN_LAYERS == 2:
	nb_hidden_nodes_1, nb_hidden_nodes_2 = 2048, 2048


print "\nnbGroups = " + str(inputdata.NUM_CLASSES)
print "nb_hidden_layers = " + str(nn.NUM_HIDDEN_LAYERS)
# print "regularization : " + str(regularization)
print "num_steps = " + str(FLAGS.num_steps)
print "num_epochs = " + str(FLAGS.num_epochs)
print "batch_size = " + str(FLAGS.batch_size) + "\n"



# ----------------------------------------------------------------------------- #
# 
# ----------------------------------------------------------------------------- #

def placeholder_inputs(batch_size, name=0):
	"""Generate placeholder variables to represent the input tensors.
	These placeholders are used as inputs by the rest of the model building
	code and will be fed from the downloaded data in the .run() loop, below.
	Args:
	batch_size: The batch size will be baked into both placeholders.
	Returns:
	images_placeholder: Images placeholder.
	labels_placeholder: Labels placeholder.
	"""
	# Note that the shapes of the placeholders match the shapes of the full
	# image and label tensors, except the first dimension is now batch_size
	# rather than the full size of the train or test data sets.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, inputdata.NUM_POINTS * inputdata.NUM_FEATURES), name=name)
	tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size, inputdata.NUM_CLASSES))
	return tf_train_dataset, tf_train_labels

# ----------------------------------------------------------------------------- #
# 																				#
# 						   Passons aux choses serieuses							#
# 																				#
# ----------------------------------------------------------------------------- #
# 
def main(_):
	train_dataset, train_labels, valid_dataset, valid_labels = get_inputs(pickle_file)
	# run_training(train_dataset, train_labels, valid_dataset, valid_labels)

	# Construct the graph
	graph = tf.Graph()
	with graph.as_default():
		# Input data.
		with tf.name_scope('Inputs_management'):
			# tf_train_dataset, tf_train_labels = placeholder_inputs(FLAGS.batch_size, name='data')
			tf_train_dataset = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, inputdata.NUM_POINTS * inputdata.NUM_FEATURES), name='tf_train_dataset')
			tf_train_labels = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, inputdata.NUM_CLASSES), name='tf_train_labels')

			keep_prob = tf.placeholder(tf.float32, name='keep_prob')

			tf_valid_dataset = tf.constant(valid_dataset, name="tf_valid_dataset")

			# tf_data = tf.Variable(tf.zeros([1,inputdata.NUM_POINTS * inputdata.NUM_FEATURES]))
			tf_data = tf.placeholder(tf.float32, shape=(1,inputdata.NUM_POINTS * inputdata.NUM_FEATURES), name="input")
			# tf_test_dataset = tf.constant(test_dataset)

		with tf.name_scope('Bias_and_weights_management'):
			weightsDict = nn.bias_weights_creation(nb_hidden_nodes_1, nb_hidden_nodes_2)	
		
		# Training computation.
		with tf.name_scope('Training_computations'):
			logits, weightsDict = nn.model(tf_train_dataset, weightsDict)
			
		with tf.name_scope('Loss_computation'):
			loss = nn.loss(logits, tf_train_labels, FLAGS.lambda_reg, weightsDict)
		
		
		with tf.name_scope('Optimization'):
			# Optimizer.
			optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
			# optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss)
		
		# tf.tensor_summary("W_fc1", weightsDict['W_fc1'])
		tf.summary.scalar("Loss", loss)
		summary_op = tf.summary.merge_all()
		saver = tf.train.Saver(weightsDict)

			
		with tf.name_scope('Predictions'):
			# Predictions for the training, validation, and test data.
			train_prediction = tf.nn.softmax(logits)
			valid_prediction = tf.nn.softmax(nn.model(tf_valid_dataset, weightsDict)[0], name="valid_prediction")

			data_pred = tf.nn.softmax(nn.model(tf_data, weightsDict)[0], name="output")
			# test_prediction = tf.nn.softmax(nn.model(tf_test_dataset, weightsDict)[0])


		# -------------------------- #
		#		Let's run it 		 #
		# -------------------------- #
		# 
		with tf.Session(graph=graph) as session:
			tf.global_variables_initializer().run()
			print("Initialized")

			# create log writer object
			writer = tf.summary.FileWriter('./train', graph=graph)

			for epoch in range(0, FLAGS.num_epochs):
				for step in range(FLAGS.num_steps):
					# Pick an offset within the training data, which has been randomized.
					# Note: we could use better randomization across epochs.
					offset = (step * FLAGS.batch_size) % (train_labels.shape[0] - FLAGS.batch_size)
					# Generate a minibatch.
					batch_data = train_dataset[offset:(offset + FLAGS.batch_size), :]
					batch_labels = train_labels[offset:(offset + FLAGS.batch_size), :]
					# Prepare a dictionary telling the session where to feed the minibatch.
					# The key of the dictionary is the placeholder node of the graph to be fed,
					# and the value is the numpy array to feed to it.
					feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.7}
					_, l, predictions, summary = session.run([optimizer, loss, train_prediction, summary_op], feed_dict=feed_dict)
					# _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)


					# write log
					batch_count = 20
					writer.add_summary(summary, epoch * batch_count + step)


					if (step % 500 == 0):
						print("Minibatch loss at step %d: %f" % (step, l))
						print("Minibatch accuracy: %.1f%%" % nn.accuracy(predictions, batch_labels)[0])
						print("Validation accuracy: %.1f%%" % nn.accuracy(valid_prediction.eval(feed_dict = {keep_prob:1.0}), valid_labels)[0])

			# finalaccuracy, mat_confusion, PPV, TPR = nn.accuracy(test_prediction.eval(feed_dict={keep_prob:1.0}), test_labels)
			# print "\n AVEC DROPOUT\n"
			# print("Test accuracy: %.1f%%" % finalaccuracy)
			# print("\n\nConfusion matrix :\n" + str(mat_confusion))
			# print "\n PPV : " + str(PPV)
			# print "\n TPR : " + str(TPR)

			save_path = saver.save(session, saveModelPath, write_meta_graph=True)
			print("Model saved in file: %s" % save_path)

			# return data_pred
		

if __name__ == '__main__':
    tf.app.run()







