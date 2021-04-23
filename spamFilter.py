#Import required libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataset = pd.read_csv('emails.csv')
dataset.head()

#remove duplicate values if there are
dataset.drop_duplicates(inplace=True)
dataset.shape

#Check for missing values
dataset.isnull().sum()

#Example
ncData = dataset.iloc[:, 1:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(ncData, ncData['Prediction'], test_size = 0.20, random_state = 0)

#get the shape of dataset
ncData.shape

#Create and train the Naive Bayes Classifier
classifier = MultinomialNB().fit(X_train, y_train)

#Print the predictions
print(classifier.predict(X_train))

#Print the actual values
print(y_train.values)

#Evaluate the model on the training data set
print("Evaluate the model on the training data set")
print()
prediction = classifier.predict(X_train)
print(classification_report(y_train, prediction))
print()
print('Confusion Matrix : \n', confusion_matrix(y_train, prediction))
print()
print('Accuracy : ', accuracy_score(y_train, prediction))
print()

#Evaluate the model on the Testing data set
print("Evaluate the model on the Testing data set")
print()
prediction = classifier.predict(X_test)
print(classification_report(y_test, prediction))
print()
print('Confusion Matrix : \n', confusion_matrix(y_test, prediction))
print()
print('Accuracy : ', accuracy_score(y_test, prediction))