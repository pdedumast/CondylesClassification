import numpy as np
import os
# import vtk
from six.moves import cPickle as pickle
import neuralnetwork as nn
import inputdata 
# ----------------------------------------------------------------------------- #


# Location for each group files
valid_train = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/surfSPHARM/5Groups-Simulated-Feat-1/"
# valid_train = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/surfSPHARM/5Groups-Feat/"

tests = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/surfSPHARM/5Groups-Feat/"
# tests = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/surfSPHARM/5Groups-Feat/"
# tests = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/"


train_size = 256
valid_size = 32
test_size = 209

train_folders = inputdata.get_folder_classes_list(valid_train)       # Folder class liste
test_folders = inputdata.get_folder_classes_list(tests)

train_datasets = inputdata.maybe_pickle(train_folders, 9)
test_datasets = inputdata.maybe_pickle(test_folders, 5)

valid_dataset, valid_labels, train_dataset, train_labels = inputdata.merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = inputdata.merge_all_datasets(test_datasets, test_size)


print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = inputdata.randomize(train_dataset, train_labels)
test_dataset, test_labels = inputdata.randomize(test_dataset, test_labels)
valid_dataset, valid_labels = inputdata.randomize(valid_dataset, valid_labels)


# ----------------------------------------------------------------------------- #
# Save the data for later reuse


pickle_file = 'condyles.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)




