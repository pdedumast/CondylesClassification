import os
import sys
from six.moves import cPickle as pickle
import neuralNetwork as nn
# import inputData
import numpy as np
import tensorflow as tf

import argparse
import zipfile
import shutil
import json




# ----------------------------------------------------------------------------- #
# 																				#
# 						   Passons aux choses serieuses							#
# 																				#
# ----------------------------------------------------------------------------- #
# 

## Reformat into a shape that's more adapted to the models we're going to train:
#   - data as a flat matrix
#   - labels as float 1-hot encodings
def reformat_data(dataset, classifier):
    dataset = dataset.reshape((-1, classifier.NUM_POINTS * classifier.NUM_FEATURES)).astype(np.float32)
    return dataset


def get_input_shape(data, classifier):
    # Get features in a matrix (NUM_FEATURES x NUM_POINTS)
    # data = input_Data.load_features(inputFile)
    data = data.reshape((-1, classifier.NUM_POINTS * classifier.NUM_FEATURES)).astype(np.float32)
    data = reformat_data(data, classifier)
    return data

# ----------------- #
# ---- RESULTS ---- #
# ----------------- #
def get_result(prediction):
    return np.argmax(prediction[0,:])


def main(_):


    # Get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputZip', action='store', dest='inputZip', help='Input zip file which contains the datasets & the parameters for the classifier', 
                        default = "")

    # parser.add_argument('-inputFile', action='store', dest='inputFile', help='Input file to classify', default = "")

    args = parser.parse_args()

    inputZip = args.inputZip
    # inputFile = args.inputFile

    basedir = os.path.dirname(inputZip)
    nameDir = os.path.splitext(os.path.basename(inputZip))[0]

    networkDir = os.path.join(basedir, nameDir)
    print "networkDir : " + networkDir

    if os.path.isdir(networkDir):
        shutil.rmtree(networkDir)
    os.mkdir(networkDir) 

    # Unpack archive
    with zipfile.ZipFile(inputZip) as zf:
        zf.extractall(basedir)

    jsonFile = os.path.join(networkDir, 'classifierInfo.json')
    saveModelPath = os.path.join(networkDir, 'CondylesClassifier')
    pickleToClassify = os.path.join(networkDir, 'toClassify.pickle')
    #
    # Create a network for the classification
    #
    with open(jsonFile) as f:    
        jsonDict = json.load(f)


    # In case our JSON file doesnt contain a valid Classifier
    if not jsonDict.has_key('CondylesClassifier'):
        print "Error: Couldn't parameterize the network."
        print "There is no 'CondylesClassifier' model."
        return 0


    # If we have the Classifier, set all parameters for the network
    classifier = nn.neuralNetwork()

    # Essential parameters
    if 'NUM_CLASSES' in jsonDict['CondylesClassifier']:
        classifier.NUM_CLASSES = jsonDict['CondylesClassifier']['NUM_CLASSES'] 
    else:
        print "Missing NUM_CLASSES"
    
    if 'NUM_POINTS' in jsonDict['CondylesClassifier']:
        classifier.NUM_POINTS = jsonDict['CondylesClassifier']['NUM_POINTS']
    else:
        print "Missing NUM_POINTS"

    if 'NUM_FEATURES' in jsonDict['CondylesClassifier']:
        classifier.NUM_FEATURES = jsonDict['CondylesClassifier']['NUM_FEATURES']
    else:
        print "Missing NUM_FEATURES"


    favorite_color = pickle.load( open( pickleToClassify, "rb" ) )

    print favorite_color

    print " .......... \n"
    for file in favorite_color.keys():
        print file 

        print "\n\n FINIIII \n\n"

        # Create session, and import existing graph
        # print shape
        myData = get_input_shape(favorite_color[file], classifier)
        session = tf.InteractiveSession()


        new_saver = tf.train.import_meta_graph(saveModelPath + '.meta')
        new_saver.restore(session, saveModelPath)
        graph = tf.Graph().as_default()

        # Get useful tensor in the graph
        tf_data = session.graph.get_tensor_by_name("Inputs_management/input:0")
        data_pred = session.graph.get_tensor_by_name("Predictions/output:0")

        feed_dict = {tf_data: myData}
        data_pred = session.run(data_pred, feed_dict=feed_dict)

        result = get_result(data_pred)
        print "Shape : " + os.path.basename(file)
        print "Group predicted :" + str(result) + "\n"

    return result



if __name__ == '__main__':
    tf.app.run()



