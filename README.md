# Machine-Learning-Exercises

## Overview
Included in this repository are some practical exercises in machine learning classification. The model itself (network.py) is a simple deep feed-forward neural network and was largely taken from [Michael Nielsen's *Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/index.html).

## Goal
The goal of this project was to learn how different hyperparameters, activation functions, and neuron density work together to optimize a cost function and how they affect model accuracy. 

## Data Sets
I trained this model to classify digits in the classic MNIST dataset of handwritten digits as well as the more difficult notMNIST dataset containing digits in many different (and difficult to read) fonts. The last dataset contains patient lifestyle information (smoking/non-smoking, obesity, etc.) and whether or not they experienced a heart attack. The model attempts to predict whether a patient will have a heart attach based on their lifestyle. This dataset only contained 139 records, making it difficult to get accurate predictions. 

## Outcomes
By tuning various hyperparamters such as the learning rate, batch size, epochs and number of neurons, I was able to attain 98% (9808/10000) accuracy on classifying MNIST digits, 93% (9308/10000) accuracy on classifying notMNIST digits, and 77% (107/139) accuracy on classifying whether a patient will have a heart attack based on their lifestyle.



