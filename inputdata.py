import os
import numpy as np
import pandas as pd
import vtk
from six.moves import cPickle as pickle



NUM_POINTS = 1002
NUM_CLASSES = 6
NUM_FEATURES = 3 + NUM_CLASSES + 4	# Normals + NUM_CLASSES + curvatures

#
# Function get_folder_classes_list(datasetPath)
# 	For a given folder, return the list of subfolders
#
def get_folder_classes_list(datasetPath):
    dataset_folders = [os.path.join(datasetPath, d) for d in sorted(os.listdir(datasetPath))]
    
    # Delete .DS_Store file if there is one
    if dataset_folders.count(str(datasetPath) + ".DS_Store"):
        dataset_folders.remove(str(datasetPath) + ".DS_Store")

    return dataset_folders


#
# Function load_features(file)
#   Load the shape stored in the filename "shape" and extract features (normals + mean distances + curvatures), stored in a 2D array (currentData)
#   Features are normalized (normals are already done, in previous program CondylesFeaturesExtractor with vtkPolyDataNormals)
#
def load_features(shape):
    dataset = np.ndarray(shape=(1, NUM_POINTS, NUM_FEATURES), dtype=np.float32)

    try:
        reader_poly = vtk.vtkPolyDataReader()
        reader_poly.SetFileName(shape)
        # print "shape : " + shape

        reader_poly.Update()
        geometry = reader_poly.GetOutput()

        if not geometry.GetNumberOfPoints() == NUM_POINTS:
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
        for i in range(0,NUM_POINTS):
            for numComponent in range(0, nbCompPosition):
                value = positionArray.GetComponent(i, numComponent)
                if value < positionMin:
                    positionMin = value
                if value > positionMax:
                    positionMax = value
        positionDepth = positionMax - positionMin

        # ***** Get distances to each mean group (nbGroups components) and normalization *****
        listGroupMean = list()
        for i in range(0, NUM_CLASSES):
            name = "distanceGroup" + str(i)
            temp = geometry.GetPointData().GetScalars(name)
            temp_range = temp.GetRange()
            temp_min, temp_max = temp_range[0], temp_range[1]
            for j in range(0,NUM_POINTS):
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
        currentData = np.ndarray(shape=(NUM_POINTS, NUM_FEATURES), dtype=np.float32)
        for i in range(0, NUM_POINTS):

            # Stock normals in currentData
            for numComponent in range(0, nbCompNormal):
                currentData[i, numComponent] = normalArray.GetComponent(i, numComponent)

            for numComponent in range(0, NUM_CLASSES):
                currentData[i, numComponent + nbCompNormal] = listGroupMean[numComponent].GetTuple1(i)

            value = 2 * (meanCurvArray.GetTuple1(i) - meanCurveMin) / meanCurveDepth -1
            currentData[i, NUM_CLASSES + nbCompNormal] = value

            value = 2 * (maxCurvArray.GetTuple1(i) - maxCurveMin) / maxCurveDepth -1
            currentData[i, NUM_CLASSES + nbCompNormal + 1] = value

            value = 2 * (minCurvArray.GetTuple1(i) - minCurveMin) / minCurveDepth -1
            currentData[i, NUM_CLASSES + nbCompNormal + 2] = value

            value = 2 * (gaussCurvArray.GetTuple1(i) - gaussCurveMin) / gaussCurveDepth -1
            currentData[i, NUM_CLASSES + nbCompNormal + 3] = value


    except IOError as e:
        print('Could not read:', shape, ':', e, '- it\'s ok, skipping.')

    # print('Full dataset tensor:', dataset.shape)
    # print('Mean:', np.mean(dataset))
    # print('Standard deviation:', np.std(dataset))
    return currentData


#
# Function load_features_classe(folder, min_num_shapes)
#   Call load_features for a entire folder/classe. Checl if there's enough shapes in a classe.
#
def load_features_classe (folder, min_num_shapes) :

	vtk_filenames = os.listdir(folder)      # Juste le nom du vtk file

	# Delete .DS_Store file if there is one
	if vtk_filenames.count(".DS_Store"):
		vtk_filenames.remove(".DS_Store")

	dataset = np.ndarray(shape=(len(vtk_filenames), NUM_POINTS, NUM_FEATURES), dtype=np.float32)

	num_shapes = 0
	for shape in vtk_filenames:
		shapePath = os.path.join(folder,shape)

		# Prepare data
		currentData = load_features(shapePath)

		# Stack the current finished data in dataset
		dataset[num_shapes, :, :] = currentData
		num_shapes = num_shapes + 1

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
            dataset = load_features_classe(folder, min_num_shapes_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


#
# Function make_arrays(nb_rows, nbPoints, nbFeatures)
#
#
def make_arrays(nb_rows, nbPoints, nbFeatures):
    if nb_rows:
        dataset = np.ndarray((nb_rows, NUM_POINTS, NUM_FEATURES), dtype=np.float32)
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
    valid_dataset, valid_labels = make_arrays(valid_size, NUM_POINTS, NUM_FEATURES)
    train_dataset, train_labels = make_arrays(train_size, NUM_POINTS, NUM_FEATURES)
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

#
# Function merge_all_datasets(pickle_files, train_size, valid_size=0)
#
#
def merge_all_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, NUM_POINTS, NUM_FEATURES)
    train_dataset, train_labels = make_arrays(train_size, NUM_POINTS, NUM_FEATURES)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, 0
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

#
# Function randomize(dataset, labels)
#   Randomize the data and their labels
#
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels




## Reformat into a shape that's more adapted to the models we're going to train:
#   - data as a flat matrix
#   - labels as float 1-hot encodings
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, NUM_POINTS * NUM_FEATURES)).astype(np.float32)
	labels = (np.arange(NUM_CLASSES) == labels[:,None]).astype(np.float32)
	return dataset, labels
