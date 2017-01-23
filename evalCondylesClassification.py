import os
import sys
from six.moves import cPickle as pickle
import neuralnetwork as nn
import inputdata
import numpy as np
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-inputFile', action='store', dest='inputFile', help='Input file to classify', 
                    default = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/surfSPHARM/5Groups-Feat/G00/34551376_Left_pp_surfSPHARM-Features.vtk")

parser.add_argument('-saveModelPath', action='store', dest='saveModelPath', help='Path to the saved model to use', default='/Users/prisgdd/Desktop/weights_5Groups-surfSPHARM.ckpt')

args = parser.parse_args()
inputFile = args.inputFile
saveModelPath = args.saveModelPath


# ----------------------------------------------------------------------------- #
# 									PROGRAM										#
# ----------------------------------------------------------------------------- #
if nn.NUM_HIDDEN_LAYERS == 1:
    nb_hidden_nodes_1 = 2048
elif nn.NUM_HIDDEN_LAYERS == 2:
    nb_hidden_nodes_1, nb_hidden_nodes_2 = 2048, 2048



# ----------------------------------------------------------------------------- #
# 																				#
# 						   Passons aux choses serieuses							#
# 																				#
# ----------------------------------------------------------------------------- #
# 


# vtk_filenames = os.listdir(inputDir)      # Juste le nom du vtk file

# # Delete .DS_Store file if there is one
# if vtk_filenames.count(".DS_Store"):
#     vtk_filenames.remove(".DS_Store")

# for shape in vtk_filenames:

#     inputFile = os.path.join(inputDir,shape)




def get_input(inputFile):
    # Get features in a matrix (NUM_FEATURES x NUM_POINTS)
    data = inputdata.load_features(inputFile)
    data = data.reshape((-1, inputdata.NUM_POINTS)).astype(np.float32)
    data = inputdata.reformat_data(data)
    return data



def run_eval(data):
    # Construct the graph
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        with tf.name_scope('Inputs_management'):
            tf_data = tf.constant(data, name="data")

        with tf.name_scope('Bias_and_weights_management'):
            weightsDict = nn.bias_weights_creation(nb_hidden_nodes_1, nb_hidden_nodes_2)    
        # Training computation.
        with tf.name_scope('Network_computations'):
            logits,weightsDict = nn.model(tf_data, weightsDict)
            
        with tf.name_scope('Predictions'):
            test_prediction = tf.nn.softmax(logits)

        with tf.name_scope('Restore_weights'):
            saver = tf.train.Saver(weightsDict)


    	with tf.Session(graph=graph) as session:
    		saver.restore(session, saveModelPath)
    		feed_dict = {tf_data: data}
    		prediction = session.run(test_prediction, feed_dict=feed_dict)
    return prediction


# ----------------- #
# ---- RESULTS ---- #
# ----------------- #
def get_result(prediction):
    pred = np.argmax(prediction[0,:])

    # if pred == 0:
    # 	result = "00"
    # elif pred == 1:
    # 	result = "01"
    # elif pred == 2:
    # 	result = "03"
    # elif pred == 3:
    # 	result = "04"
    # elif pred == 4:
    # 	result = "05"
    # elif pred == 5:
    # 	result = "06-07"
    # return result
    return pred


def main(_):
    data = get_input(inputFile)
    prediction = run_eval(data)
    result = get_result(prediction)
    print "Shape : " + os.path.basename(inputFile)
    print "Group predicted :" + str(result) + "\n"
    return result


if __name__ == '__main__':
    tf.app.run()



