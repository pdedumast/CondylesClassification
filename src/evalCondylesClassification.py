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

parser.add_argument('-saveModelPath', action='store', dest='saveModelPath', help='Path to the saved model to use', default='./weights_5Groups')

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
    		feed_dict = {tf_valid_dataset: data}
    		prediction = session.run(test_prediction, feed_dict=feed_dict)
    return prediction


# ----------------- #
# ---- RESULTS ---- #
# ----------------- #
def get_result(prediction):
    return np.argmax(prediction[0,:])


def main(_):

    # Create session, and import existing graph
    myData = get_input(inputFile)
    session = tf.InteractiveSession()
    new_saver = tf.train.import_meta_graph('./weights_5Groups.meta')
    new_saver.restore(session, './weights_5Groups')
    graph = tf.Graph().as_default()

    # listTensor =[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    
    # Get useful tensor in the graph
    # tf_data = session.graph.get_collection(name="input:0", scope="Inputs_management") 
    tf_data = session.graph.get_tensor_by_name("Inputs_management/input:0")
    data_pred = session.graph.get_tensor_by_name("Predictions/output:0")

    feed_dict = {tf_data: myData}
    data_pred = session.run(data_pred, feed_dict=feed_dict)
    
    result = get_result(data_pred)
    print "Shape : " + os.path.basename(inputFile)
    print "Group predicted :" + str(result) + "\n"
    return result


if __name__ == '__main__':
    tf.app.run()



