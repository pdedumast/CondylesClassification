import numpy as np
import os
import tensorflow as tf
from six.moves import cPickle as pickle

import vtk


# --------------------------------------------------------------------------------------------------- #

nbPoints = 1002
nbGroups = 6

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
    nbFeatures = 3 + nbGroups
elif featuresType == "norm-dist-curv":
    nbFeatures = 3 + nbGroups + 4
elif featuresType == "norm-curv":
    nbFeatures = 3 + 4 
# --------------------------------------------------------------------------------------------------- #

# Location for each group files
if nbGroups == 8:
    valid_train = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/7Groups-Feat/"
elif nbGroups == 6:
    # valid_train = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/5Groups-Feat/"
    valid_train = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/5Groups-Limane-Feat/"

train_folders = [os.path.join(valid_train, d) for d in sorted(os.listdir(valid_train))]       # Folder class liste

# Delete .DS_Store file if there is one
if train_folders.count(str(valid_train) + ".DS_Store"):
    train_folders.remove(str(valid_train) + ".DS_Store")

# test_folders = train_folders
tests = "/Users/prisgdd/Documents/Projects/CNN/DataPriscille/5Groups-Feat/"
test_folders = [os.path.join(tests, d) for d in sorted(os.listdir(tests))]
# Delete .DS_Store file if there is one
if test_folders.count(str(test_folders) + ".DS_Store"):
    test_folders.remove(str(test_folders) + ".DS_Store")

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

            if geometry.GetNumberOfPoints() < nbPoints or geometry.GetNumberOfPoints() > nbPoints:
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

            # ***** Get distances to each mean group (nbGroups components) and normalization *****
            listGroupMean = list()
            for i in range(0, nbGroups):
                name = "distanceGroup" + str(i)
                temp = geometry.GetPointData().GetScalars(name)
                temp_range = temp.GetRange()
                temp_min = temp_range[0]
                temp_max = temp_range[1]
                for j in range(0,nbPoints):
                    temp.SetTuple1(j, 2 * (temp.GetTuple1(j) - temp_min) / (temp_max) - 1)
                listGroupMean.append(temp)

            # ***** Get Curvatures and value for normalization (4 components) *****
            meanCurvName = "Mean_Curvature"
            meanCurvArray = geometry.GetPointData().GetScalars(meanCurvName)
            meanCurveRange = meanCurvArray.GetRange()
            meanCurveMin = meanCurveRange[0]
            meanCurveMax = meanCurveRange[1]
            meanCurveDepth = meanCurveMax - meanCurveMin

            maxCurvName = "Maximum_Curvature"
            maxCurvArray = geometry.GetPointData().GetScalars(maxCurvName)
            maxCurveRange = maxCurvArray.GetRange()
            maxCurveMin = maxCurveRange[0]
            maxCurveMax = maxCurveRange[1]
            maxCurveDepth = maxCurveMax - maxCurveMin

            minCurvName = "Minimum_Curvature"
            minCurvArray = geometry.GetPointData().GetScalars(minCurvName)
            minCurveRange = minCurvArray.GetRange()
            minCurveMin = minCurveRange[0]
            minCurveMax = minCurveRange[1]
            minCurveDepth = minCurveMax - minCurveMin

            gaussCurvName = "Gauss_Curvature"
            gaussCurvArray = geometry.GetPointData().GetScalars(gaussCurvName)
            gaussCurveRange = gaussCurvArray.GetRange()
            gaussCurveMin = gaussCurveRange[0]
            gaussCurveMax = gaussCurveRange[1]
            gaussCurveDepth = gaussCurveMax - gaussCurveMin

            # For each point of the current shape
            currentData = np.ndarray(shape=(nbPoints, nbFeatures), dtype=np.float32)
            for i in range(0, nbPoints):

                if featuresType == "norm-pos":      # [Nx, Ny, Nz, Px, Py, Pz]
                    # Stock position in currentData - Normalization
                    for numComponent in range(0, nbCompPosition):
                        value = positionArray.GetComponent(i, numComponent)
                        currentData[i, numComponent] = 2 * (value - positionMin) / positionDepth - 1

                    # Stock normals in currentData
                    for numComponent in range(0, nbCompNormal):
                        currentData[i, numComponent + nbCompPosition] = normalArray.GetComponent(i, numComponent)
                
                elif featuresType == "norm":        # [Nx, Ny, Nz]
                    # Stock normals in currentData
                    for numComponent in range(0, nbCompNormal):
                        currentData[i, numComponent] = normalArray.GetComponent(i, numComponent)

                elif featuresType == "norm-dist":   # [Nx, Ny, Nz, D0, ... Dmax]
                    # Stock normals in currentData
                    for numComponent in range(0, nbCompNormal):
                        currentData[i, numComponent] = normalArray.GetComponent(i, numComponent)

                    for numComponent in range(0, nbGroups):
                        currentData[i, numComponent + nbCompNormal] = listGroupMean[numComponent].GetTuple1(i)

                elif featuresType == "norm-dist-curv":  # [Nx, Ny, Nz, D0, ... Dmax, meanCurv, maxCurv, minCurv, GaussCurv]
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

                elif featuresType == "norm-curv":       # [Nx, Ny, Nz, meanCurv, maxCurv, minCurv, GaussCurv]
                    # Stock normals in currentData
                    for numComponent in range(0, nbCompNormal):
                        currentData[i, numComponent] = normalArray.GetComponent(i, numComponent)

                    # Stock mean curvature in currentData
                    value = 2 * (meanCurvArray.GetTuple1(i) - meanCurveMin) / meanCurveDepth -1
                    currentData[i, nbCompNormal] = value

                    # Stock max curvature in currentData
                    value = 2 * (maxCurvArray.GetTuple1(i) - maxCurveMin) / maxCurveDepth -1
                    currentData[i, nbCompNormal + 1] = value

                    # Stock min curvature in currentData
                    value = 2 * (minCurvArray.GetTuple1(i) - minCurveMin) / minCurveDepth -1
                    currentData[i, nbCompNormal + 2] = value

                    # Stock gaussian curvature in currentData
                    value = 2 * (gaussCurvArray.GetTuple1(i) - gaussCurveMin) / gaussCurveDepth -1
                    currentData[i, nbCompNormal + 3] = value

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

train_datasets = maybe_pickle(train_folders, 9)
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
    # end_l = tsize_per_class
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


def merge_all_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, nbPoints, nbFeatures)
    train_dataset, train_labels = make_arrays(train_size, nbPoints, nbFeatures)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, 0
    # end_l = vsize_per_class + tsize_per_class
    # end_l = tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                shape_set = pickle.load(f)
                print shape_set.shape
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(shape_set)
                if valid_dataset is not None:
                    valid_shapes = shape_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_shapes
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                tsize_current_class = shape_set.shape[0]
                end_t += tsize_current_class - vsize_per_class
                end_l = tsize_current_class
                train_shapes = shape_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_shapes
                train_labels[start_t:end_t] = label
                start_t += tsize_current_class - vsize_per_class
                # end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels




# --------------------------------------------------------------------------------------------------- #

if nbGroups == 8:  
    train_size = 64
    valid_size = 8
    test_size = 45
elif nbGroups == 6:  
    train_size = 544
    valid_size = 32
    test_size = 209


valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_all_datasets(test_datasets, test_size)

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




