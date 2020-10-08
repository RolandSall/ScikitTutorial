from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
# split features and labels
firstFeature = iris.data
secondFeature = iris.target

# to check the data
# print(firstFeature, secondFeature)

# to check the skeleton on of the data
# print(firstFeature.shape, secondFeature.shape)

# test_size is used to determine how much data we want to use for training
x_train, x_test, y_train, y_test = train_test_split(firstFeature, secondFeature, test_size=0.2)
# print(x_train.shape, x_test.shape)


