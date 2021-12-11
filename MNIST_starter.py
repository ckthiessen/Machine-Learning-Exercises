import numpy as np
import idx2numpy
import network

MAX_PIXEL_VALUE = 255
FLATTENED_SIZE = 784

# converts a 1d python list into a (1,n) row vector
def rv(vec):
    return np.array([vec])
    
# converts a 1d python list into a (n,1) column vector
def cv(vec):
    return rv(vec).T
        
# creates a (size,1) array of zeros, whose ith entry is equal to 1    
def onehot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return cv(vec)


##################################################
# NOTE: make sure these paths are correct for your directory structure

# training data
trainingImageFile = "data/train-images.idx3-ubyte"
trainingLabelFile = "data/train-labels.idx1-ubyte"

# testing data
testingImageFile = "data/t10k-images.idx3-ubyte"
testingLabelFile = "data/t10k-labels.idx1-ubyte"


# returns the number of entries in the file, as well as a list of integers
# representing the correct label for each entry
def getLabels(labelfile):
    file = open(labelfile, 'rb')
    file.read(4)
    n = int.from_bytes(file.read(4), byteorder='big') # number of entries
    
    labelarray = bytearray(file.read())
    labelarray = [b for b in labelarray]    # convert to ints
    file.close()
    
    return n, labelarray

# returns a list containing the pixels for each image, stored as a (784, 1) numpy array
def getImgData(imagefile):
    # returns an array whose entries are each (28x28) pixel arrays with values from 0 to 255.0
    images = idx2numpy.convert_from_file(imagefile) 
    print(f'{images.shape = }')
    
    # We want to flatten each image from a 28 x 28 to a 784 x 1 numpy array
    flattened_images = images.reshape(images.shape[0], FLATTENED_SIZE, 1)
    print(f'{flattened_images.shape = }')
    
    # convert to floats in [0,1] (only really necessary if you have other features, but we'll do it anyways)
    features = flattened_images / MAX_PIXEL_VALUE
   
    return features


# reads the data from the four MNIST files,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    ntrain, train_labels = getLabels(trainingLabelFile)
    ntest, test_labels = getLabels(testingLabelFile)

    print(f'{ntrain = }')
    print(f'{ntest = }')

    one_hot_training_labels = [onehot(label, 10) for label in train_labels]

    training_imgs = getImgData(trainingImageFile)
    testing_imgs = getImgData(testingImageFile)

    training_data = zip(training_imgs, one_hot_training_labels)
    testing_data = zip(testing_imgs, test_labels)

    return (training_data, testing_data)

###################################################


trainingData, testingData = prepData()

net = network.Network([784, 64, 64, 10])
net.SGD(trainingData, 5, 20, 7, 'part1_improved', 0.95, test_data = testingData)

