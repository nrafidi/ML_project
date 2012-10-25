#This module has two methods: one to train a ridge regressor and one
#to use a given ridge regressor to make predictions. If y is of dimension k
#you must call this k times to make k separate regressors

#Requires Numpy, Scipy, and Scikit-learn
import numpy as np
import sklearn.linear_model

#features = array-like object, size n_samples x n_features
#labels = array-like object, size n_samples x 1
#centered = boolean expressing whether the data has been centered. If set to
#true, the model will not calculate an intercept
#alphas =  regularization parameter to iterate over in the CV
#classifier = A ridge classifier
def lin_reg_train(features, labels, not_centered):
    alphas = np.array([0.1, 1, 10, 100])
    classifier = sklearn.linear_model.RidgeClassifierCV(alphas, not_centered)
    classifier.fit(features, labels)
    return classifier

#Given a classifier and a feature array, what is the prediction
#This is probably not necessary
def lin_reg_test(classifier, features):
    return classifier.predict(features)
    
