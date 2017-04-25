# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 14:01:13 2016

@author: Rashmita Rout
"""
###################################IMPORT PACKAGES#############################
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
###############################################################################

##############################ABOUT THE DATASET################################
#The MNIST(Mixed National Institute of Standards and Technology) dataset 
#is used in this project. All the digit images have been pre-processed 
#such that the digit is centered on a 28 × 28 block of 8-bit grayvalues. 
############################IMPORT TRAIN & TEST FILES##########################

#first we must begin with importing the train and test dataset
train = pd.read_csv("C:/Users/Rashmita Rout/Desktop/Digit Data/train.csv")
test = pd.read_csv("C:/Users/Rashmita Rout/Desktop/Digit Data/test.csv")

#converting the test and train datasets into dataframes
train_df= pd.DataFrame(train)
test_df= pd.DataFrame(test)

#we check both the datasets to find the columns present in them 
#and then find the number of missing values in each column
train_df.info()
test_df.info()

#We see that our train dataset has 42000 rows and 785 columns 
#and the test dataset has 28000 rows and 784 columns 
#there are no missing values in this dataset.
###############################################################################

##########################DATA PRE_PROCESSING##################################
#########################CREATE TRAIN & TEST FEATURES AND LABELS###############
#In Python we need to divide the data set into features and labels
#First, we will divide the train data to features and labels
#The train features will only have the predictor columns,i.e., 
#Removing the dependent variable "label" from the training data usin drop()
train_features = train_df.drop(['label'],axis=1)

#Now storing the dependent variable "Label" as train labels
train_labels = train.label

#saving the test data as the test features
test_features = test_df
###############################################################################

#########################FEATURE EXTRACTION####################################
########################PRINCIPAL COMPONENT ANALYSIS###########################
#PCA is an unsupervised method for reducing the dimensionality of the existing 
#data set and extracting important information.
#We are selecting only 50 components
pca = PCA(n_components=50,whiten=True)
#fitting the train features
pca.fit(train_features)
#Transforming the train features and storing it as pca_train
pca_train = pca.transform(train_features)
#Transforming the test features and storing it as pca_test
pca_test = pca.transform(test_features)
###############################################################################

########################LINEAR SUPPORT VECTOR MACHINE##########################
###############This gave the best score########################################
#Initialize the classifier
clf =SVC(kernel='rbf', C=1000, gamma=0.001)
#Fit the model
clffit = clf.fit(pca_train,train_labels)
#Predict the test labels
pred = clf.predict(pca_test)

#generating the submission file
#importing the predicted values into excel file in two columns "ImageId" and "label" 
final= pd.DataFrame()
final['ImageId']= range(1,28001,1)
final['label']= pred

#submitting the predicted values in a csv file
final.to_csv("C:/Users/Rashmita Rout/Desktop/Digit Data/Result_4.csv",index=False)
###############################################################################

####################RANDOM FOREST CLASSIFIER###################################
#Initialize the classifier
clf= RandomForestClassifier(n_estimators=1000)
#Fit the model
clffit = clf.fit(pca_train,train_labels)
#Predict the test labels
pred= clf.predict(pca_test)

#generating the submission file
#importing the predicted values into excel file in two columns "ImageId" and "label" 
final= pd.DataFrame()
final['ImageId']= range(1,28001,1)
final['label']= pred

#submitting the predicted values in a csv file
final.to_csv("C:/Users/Rashmita Rout/Desktop/Digit Data/Result_3.csv",index=False)
###############################################################################

############################MY LEARNINGS####################################### 
#The SVM seems to be particularly well suited for digit recognition as it 
#has the the best performance when compared to random forest or KNN.
#SVM is not only accurate but also faster than the other methods

###############################################################################

