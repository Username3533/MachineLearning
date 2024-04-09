""" Learning Machine Learning/Deep Learning/Neural Networks """

import array
import asyncio
import argparse
import sqlite3
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from cryptography.fernet import Fernet
from selenium import webdriver
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical


""" Trains a decision tree model using iris data set from scikit, predicts 3 new types of iris based on given data """

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

new_data = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [7.7, 3.8, 6.7, 2.2]]
new_predictions = clf.predict(new_data)

print('New data:', new_data)
print('New predictions:', new_predictions)

