import numpy as np
import os
import tensorflow as tf
from six.moves import cPickle as pickle

import vtk


# --------------------------------------------------------------------------------------------------- #

# Location for each group files
location = "/Users/prisgdd/Documents/Projects/CNN/DataOutput/"
train_folders = [os.path.join(location, d) for d in sorted(os.listdir(location))]       # Folder class liste

# Delete .DS_Store file if there is one
if train_folders.count(str(location) + ".DS_Store"):
    train_folders.remove(str(location) + ".DS_Store")

test_folders = train_folders

nbPoints = 1000  
nbFeatures = 6
pixel_depth = 255.0  # Number of levels per pixel.


# --------------------------------------------------------------------------------------------------- #

#
# Function load_features(folder, min_num_shapes)
#   Load each shape of each class folder, and extract features (normals + postition), stored in a 2D array (currentData)
#   All 2D array containing features are stacked together in a 3D array (shape index, nbPoints, nbFeatures) of floating point values
#   Features are normalized (normals are already done, in previous program CondylesFeaturesExtractor)
#
def load_features(folder, min_num_shapes):
    """Load the data for one class of condyles."""
    vtk_filenames = os.listdir(folder)      # Juste le nom du vtk file

    print "folder : " + folder
    


    # Delete .DS_Store file if there is one
    if vtk_filenames.count(".DS_Store"):
        vtk_filenames.remove(".DS_Store")

    dataset = np.ndarray(shape=(len(vtk_filenames), nbPoints, nbFeatures), dtype=np.float32)

    num_shapes = 0
    for shape in vtk_filenames:
        shape = os.path.join(folder, shape)     # PATH depuis ~/ + nom du vtk file

        try:
            reader_poly = vtk.vtkPolyDataReader()
            reader_poly.SetFileName(shape)
            # print "shape : " + shape

            reader_poly.Update()
            geometry = reader_poly.GetOutput()

            if geometry.GetNumberOfPoints() < nbPoints:
                raise Exception('Unexpected number of points in the shape: %s' % str(geometry.GetNumberOfPoints()))

            # --------------------------------- #
            # ----- GET ARRAY OF FEATURES ----- #
            # --------------------------------- #

            # ***** Get positions (3 useful components) *****
            positionName = "position"
            positionArray = geometry.GetPointData().GetVectors(positionName)
            nbCompPosition = positionArray.GetElementComponentSize() - 1  # -1 car 4eme comp = 1ere du pt suivant

            # Get position range
            positionRange = positionArray.GetRange()
            positionMin = 1000000
            positionMax = -1000000
            for i in range(0,nbPoints):
                for numComponent in range(0, nbCompPosition):
                    value = positionArray.GetComponent(i, numComponent)
                    if value < positionMin:
                        positionMin = value
                    if value > positionMax:
                        positionMax = value
            positionDepth = positionMax - positionMin

            # ***** Get normals (3 useful components) *****
            normalArray = geometry.GetPointData().GetNormals()
            nbCompNormal = normalArray.GetElementComponentSize() - 1  # -1 car 4eme comp = 1ere du pt suivant

            # For each point of the current shape
            currentData = np.ndarray(shape=(nbPoints, nbFeatures), dtype=np.float32)
            for i in range(0, nbPoints):

                # Stock position in currentData - Normalization
                for numComponent in range(0, nbCompPosition):
                    value = positionArray.GetComponent(i, numComponent)
                    currentData[i, numComponent] = 2 * (value - positionMin) / positionDepth - 1

                # Stock normals in currentData
                for numComponent in range(0, nbCompNormal):
                    currentData[i, numComponent + nbCompPosition] = normalArray.GetComponent(i, numComponent)

            # Stack the current finished data in dataset
            dataset[num_shapes, :, :] = currentData
            num_shapes = num_shapes + 1

        except IOError as e:
            print('Could not read:', shape, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_shapes, :, :]
    if num_shapes < min_num_shapes:
        raise Exception('Many fewer images than expected: %d < %d' %(num_shapes, min_num_shapes))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    print ""
    return dataset


#
# Function maybe_pickle(data_folders, min_num_shapes_per_class, force=False)
#   Pickle features array sorted by class
#
def maybe_pickle(data_folders, min_num_shapes_per_class, force=False):
    dataset_names = []
    folders = list()
    for d in data_folders:
        if os.path.isdir(os.path.join(data_folders, d)):
            folders.append(os.path.join(data_folders, d))
    for folder in folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_features(folder, min_num_shapes_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

# --------------------------------------------------------------------------------------------------- #

train_datasets = maybe_pickle(train_folders, 10)
test_datasets = maybe_pickle(test_folders, 5)


# --------------------------------------------------------------------------------------------------- #

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

#
# Function merge_datasets(pickle_files, train_size, valid_size=0)
#
#
def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, nbPoints, nbFeatures)
    train_dataset, train_labels = make_arrays(train_size, nbPoints, nbFeatures)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                shape_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(shape_set)
                if valid_dataset is not None:
                    valid_shapes = shape_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_shapes
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_shapes = shape_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_shapes
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

# --------------------------------------------------------------------------------------------------- #

train_size = 35
valid_size = 10
test_size = 10

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
#


# --------------------------------------------------------------------------------------------------- #


#
# Function randomize(dataset, labels)
#   Randomize the data and their labels
#
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


# --------------------------------------------------------------------------------------------------- #
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




