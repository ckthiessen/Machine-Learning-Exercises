import numpy as np
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

def flatten_and_normalize(features):
    return features.reshape(features.shape[0], FLATTENED_SIZE, 1) / MAX_PIXEL_VALUE


#################################################################

# reads the data from the notMNIST.npz file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    # loads the four arrays specified.
    # train_features and test_features are arrays of (28x28) pixel values from 0 to 255.0
    # train_labels and test_labels are integers from 0 to 9 inclusive, representing the letters A-J
    with np.load("data/notMNIST.npz", allow_pickle=True) as f:
        train_features, train_labels = f['x_train'], f['y_train']
        test_features, test_labels = f['x_test'], f['y_test']
        
    # need to rescale, flatten, convert training labels to one-hot, and zip appropriate components together
    flattened_normalized_train = flatten_and_normalize(train_features)
    flattened_normalized_test = flatten_and_normalize(test_features)
    
    one_hot_training_labels = [onehot(label, 10) for label in train_labels]

    training_data = zip(flattened_normalized_train, one_hot_training_labels)
    testing_data = zip(flattened_normalized_test, test_labels)
       
    return (training_data, testing_data)
    
###################################################################


trainingData, testingData = prepData()

net = network.Network([784, 64, 64, 10])
net.SGD(trainingData, 5, 20, 7, 'part2_improved', 0.90, test_data = testingData)






