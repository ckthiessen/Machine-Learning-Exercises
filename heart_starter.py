import csv
import numpy as np
import pandas as pd
import network


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

# given a data point, mean, and standard deviation, returns the z-score
def standardize(x, mu, sigma):
    return ((x - mu)/sigma)
    

##############################################

# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple
def readData(filename):

    csv_df = pd.read_csv(filename)
    csv_df['famhist'] = csv_df['famhist'].apply(lambda famhist: 1 if famhist == 'Present' else 2)
    csv_df['age'] = csv_df['age'] / csv_df['age'].max()
    for col in csv_df[['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol']]:
        mean = csv_df[col].mean()
        std_dev = csv_df[col].std()
        csv_df[col] = csv_df[col].apply(lambda x: standardize(x, mean, std_dev))

    n = csv_df['chd'].size

    labels = csv_df['chd']
    csv_df.drop(['chd', 'row.names'], axis=1, inplace=True)

    features = []
    for _, row in csv_df.iterrows():
        features.append(np.array([[x] for x in row.values]))


    return n, features, labels

################################################

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData(ratio = 0.7):

    n, features, labels = readData('data/heart.csv')

    n_train = int(n * ratio)

    training_features = features[:n_train]
    training_labels = [onehot(label, 2) for label in labels[:n_train]]    # training labels should be in onehot form

    testing_features = features[n_train:]
    testing_labels = labels[n_train:]

    training_data = zip(training_features, training_labels)
    testing_data = zip(testing_features, testing_labels)

    return (training_data, testing_data)

###################################################


trainingData, testingData = prepData()

net = network.Network([9, 16, 2])
accuracy = net.SGD(trainingData, 15, 10, 0.7, 'part3_improved', 0.76, test_data = testingData)