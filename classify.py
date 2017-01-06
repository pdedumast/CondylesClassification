import os
import sys
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import vtk

nbPoints = 1002
nbGroups = 6
nb_hidden_layers = 3
nbFeatures = 3 + nbGroups + 4

# ----------------------------------------------------------------------------- #
#
# Function load_features(file)
#   Load the shape and extract features (normals + mean distances + curvatures), stored in a 2D array (currentData)
#   Features are normalized (normals are already done, in previous program CondylesFeaturesExtractor)
#
def load_features(shape):
    dataset = np.ndarray(shape=(1, nbPoints, nbFeatures), dtype=np.float32)

    try:
        reader_poly = vtk.vtkPolyDataReader()
        reader_poly.SetFileName(shape)
        # print "shape : " + shape

        reader_poly.Update()
        geometry = reader_poly.GetOutput()

        if not geometry.GetNumberOfPoints() == nbPoints:
            raise Exception('Unexpected number of points in the shape: %s' % str(geometry.GetNumberOfPoints()))

        # --------------------------------- #
        # ----- GET ARRAY OF FEATURES ----- #
        # --------------------------------- #

        # *****
        # ***** Get normals (3 useful components) - already normalized *****
        normalArray = geometry.GetPointData().GetNormals()
        nbCompNormal = normalArray.GetElementComponentSize() - 1  # -1 car 4eme comp = 1ere du pt suivant

        # *****
        # ***** Get positions (3 useful components) *****
        positionName = "position"
        positionArray = geometry.GetPointData().GetVectors(positionName)
        nbCompPosition = positionArray.GetElementComponentSize() - 1  # -1 car 4eme comp = 1ere du pt suivant

        # Get position range & normalize
        positionMin, positionMax = 1000000, -1000000
        for i in range(0,nbPoints):
            for numComponent in range(0, nbCompPosition):
                value = positionArray.GetComponent(i, numComponent)
                if value < positionMin:
                    positionMin = value
                if value > positionMax:
                    positionMax = value
        positionDepth = positionMax - positionMin

        # ***** Get distances to each mean group (nbGroups components) and normalization *****
        listGroupMean = list()
        for i in range(0, nbGroups):
            name = "distanceGroup" + str(i)
            temp = geometry.GetPointData().GetScalars(name)
            temp_range = temp.GetRange()
            temp_min, temp_max = temp_range[0], temp_range[1]
            for j in range(0,nbPoints):
                temp.SetTuple1(j, 2 * (temp.GetTuple1(j) - temp_min) / (temp_max) - 1)
            listGroupMean.append(temp)

        # ***** Get Curvatures and value for normalization (4 components) *****
        meanCurvName = "Mean_Curvature"
        meanCurvArray = geometry.GetPointData().GetScalars(meanCurvName)
        meanCurveRange = meanCurvArray.GetRange()
        meanCurveMin, meanCurveMax = meanCurveRange[0], meanCurveRange[1]
        meanCurveDepth = meanCurveMax - meanCurveMin

        maxCurvName = "Maximum_Curvature"
        maxCurvArray = geometry.GetPointData().GetScalars(maxCurvName)
        maxCurveRange = maxCurvArray.GetRange()
        maxCurveMin, maxCurveMax = maxCurveRange[0], maxCurveRange[1]
        maxCurveDepth = maxCurveMax - maxCurveMin

        minCurvName = "Minimum_Curvature"
        minCurvArray = geometry.GetPointData().GetScalars(minCurvName)
        minCurveRange = minCurvArray.GetRange()
        minCurveMin, minCurveMax = minCurveRange[0], minCurveRange[1]
        minCurveDepth = minCurveMax - minCurveMin

        gaussCurvName = "Gauss_Curvature"
        gaussCurvArray = geometry.GetPointData().GetScalars(gaussCurvName)
        gaussCurveRange = gaussCurvArray.GetRange()
        gaussCurveMin, gaussCurveMax = gaussCurveRange[0], gaussCurveRange[1]
        gaussCurveDepth = gaussCurveMax - gaussCurveMin

        # For each point of the current shape
        currentData = np.ndarray(shape=(nbPoints, nbFeatures), dtype=np.float32)
        for i in range(0, nbPoints):

            # Stock normals in currentData
            for numComponent in range(0, nbCompNormal):
                currentData[i, numComponent] = normalArray.GetComponent(i, numComponent)

            for numComponent in range(0, nbGroups):
                currentData[i, numComponent + nbCompNormal] = listGroupMean[numComponent].GetTuple1(i)

            value = 2 * (meanCurvArray.GetTuple1(i) - meanCurveMin) / meanCurveDepth -1
            currentData[i, nbGroups + nbCompNormal] = value

            value = 2 * (maxCurvArray.GetTuple1(i) - maxCurveMin) / maxCurveDepth -1
            currentData[i, nbGroups + nbCompNormal + 1] = value

            value = 2 * (minCurvArray.GetTuple1(i) - minCurveMin) / minCurveDepth -1
            currentData[i, nbGroups + nbCompNormal + 2] = value

            value = 2 * (gaussCurvArray.GetTuple1(i) - gaussCurveMin) / gaussCurveDepth -1
            currentData[i, nbGroups + nbCompNormal + 3] = value


        # Stack the current finished data in dataset
        dataset[0, :, :] = currentData

    except IOError as e:
        print('Could not read:', shape, ':', e, '- it\'s ok, skipping.')

    # print('Full dataset tensor:', dataset.shape)
    # print('Mean:', np.mean(dataset))
    # print('Standard deviation:', np.std(dataset))
    # print ""
    return dataset



## Reformat into a shape that's more adapted to the models we're going to train:
#   - data as a flat matrix
#   - labels as float 1-hot encodings
def reformat(dataset):
	dataset = dataset.reshape((-1, nbPoints * nbFeatures)).astype(np.float32)
	return dataset



# ----------------------------------------------------------------------------- #
# 									PROGRAM										#
# ----------------------------------------------------------------------------- #



if nbGroups == 6:
	saveModelPath = 'weights_5Groups-surfSPHARM-simu-2.ckpt'
else:
	saveModelPath = 'weights_7Groups.ckpt'

if nb_hidden_layers == 2:
	nb_hidden_nodes_1 = 2048
elif nb_hidden_layers == 3:
	nb_hidden_nodes_1 = 2048
	nb_hidden_nodes_2 = 2048

# ---------------------------------------------------------------------------- #


inputFile = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/surfSPHARM/5Groups-Feat/G04/C6R_pp_surfSPHARM-Features.vtk"

# Prepare data
data = load_features(inputFile)
data = data.reshape((-1, nbFeatures * nbPoints)).astype(np.float32)
data = reformat(data)

# ----------------------------------------------------------------------------- #
# 																				#
# 						   Passons aux choses serieuses							#
# 																				#
# ----------------------------------------------------------------------------- #
# 

graph = tf.Graph()
with graph.as_default():
    # Input data.
	tf_test_dataset = tf.constant(data)

	def weight_variable(shape, name=None):
		"""Create a weight variable with appropriate initialization."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, name=name)

	def bias_variable(shape, name=None):
		"""Create a bias variable with appropriate initialization."""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial,name = name)

	if nb_hidden_layers == 1:
		W_fc1 = weight_variable([nbPoints * nbFeatures, nbGroups], "W_fc1")
		b_fc1 = bias_variable([nbGroups],"b_fc1")

	elif nb_hidden_layers == 2:
		W_fc1 = weight_variable([nbPoints * nbFeatures, nb_hidden_nodes_1], "W_fc1")
		b_fc1 = bias_variable([nb_hidden_nodes_1],"b_fc1")

		W_fc2 = weight_variable([nb_hidden_nodes_1, nbGroups], "W_fc2")
		b_fc2 = bias_variable([nbGroups],"b_fc2")

	elif nb_hidden_layers == 3:
		W_fc1 = weight_variable([nbPoints * nbFeatures, nb_hidden_nodes_1], "W_fc1")
		b_fc1 = bias_variable([nb_hidden_nodes_1],"b_fc1")

		W_fc2 = weight_variable([nb_hidden_nodes_1, nb_hidden_nodes_2], "W_fc2")
		b_fc2 = bias_variable([nb_hidden_nodes_2],"b_fc2")

		W_fc3 = weight_variable([nb_hidden_nodes_2, nbGroups], "W_fc3")
		b_fc3 = bias_variable([nbGroups],"b_fc3")

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
		feed_dict = {tf_test_dataset: data}
		prediction = session.run(test_prediction, feed_dict=feed_dict)


# ----------------- #
# ---- RESULTS ---- #
# ----------------- #

pred = np.argmax(prediction[0,:])

if pred == 0:
	result = "00"
elif pred == 1:
	result = "01"
elif pred == 2:
	result = "03"
elif pred == 3:
	result = "04"
elif pred == 4:
	result = "05"
elif pred == 5:
	result = "06-07"
filename =  os.path.basename(inputFile)
print "Shape : " + filename
print "Group predicted :" + result


