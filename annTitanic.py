# Artificial Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')

#exploring dataset
#(column index, name, non-null count, length, data type & more)
print (dataset.info())
#sample of data
dataset.sample(10)
#description of count and stats per column
#bignumbers = data_train.describe(include = 'all')

# finding null data in both sets (count per column)
print('Train columns with null values:\n', dataset.isnull().sum())
print("-"*10)

###COMPLETING: complete or delete missing values in train and test/validation dataset
## ca ne marche pas si on est en numpy array, il faut tranformer en dataframe
# ex: dataset = pd.DataFrame(data=dataset, index = ["dataset.rows"], columns = ["dataset.columns"] )
for data in dataset:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    #complete embarked with mode(highest frequency value)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
# finding null data in both sets (count per column)
print('Train columns with null values:\n', dataset.isnull().sum())
print("-"*10)

###CREATE: Feature Engineering for train and test/validation dataset
for data in dataset:    
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    #initialize to yes/1 is alone
    dataset['IsAlone'] = 1 
    # now update to no/0 if family size is greater than 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
    
    dataset['ParentChildren'] = 0
    dataset['ParentChildren'].loc[dataset['Parch']>1]=1
    #split title from name
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
   
    # #Continuous variable bins; qcut(same frquency of samples, different spacing) vs cut(different frquencyof samples, same spacing)
    # #qcut(qty cut) cut(interval cut)
    # dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    # #Age Bins/Buckets
    # dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

#group rare title names
print(dataset['Title'].value_counts())
# 10 :common minimum in statistics
title_names = (dataset['Title'].value_counts() < 10) 
#this will create a true false series with title name as index

#apply and lambda functions to find and replace
#x = every record in data1["Title"] becomes misc if ...
dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(dataset['Title'].value_counts())
print("-"*10)


#delete the cabin feature/column and others previously stated to exclude in train dataset
dataset.drop(['Name','Parch','SibSp', 'PassengerId','Cabin', 'Ticket'], axis=1, inplace = True)
## to preview data again
# dataset.info()
# dataset.sample(10)
todummy = ['Pclass', 'Sex', 'Embarked', 'Title']
for feature in todummy:  
    dummies = pd.get_dummies(dataset[feature], prefix = feature, drop_first= True)
    dataset = pd.concat([dataset, dummies], axis = 1)
    dataset.drop([feature], axis=1, inplace = True)



# dataset.columns to copy paste
#https://pbpython.com/selecting-columns.html
X = dataset.iloc[:, 1: ].values
y = dataset.iloc[:, 0].values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Applying Kernel PCA
# from sklearn.decomposition import KernelPCA
# kpca = KernelPCA(n_components = None, kernel = 'rbf')
# X_train = kpca.fit_transform(X_train)
# X_test = kpca.transform(X_test)

# # Applying LDA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# lda = LDA(n_components = None)
# X_train = lda.fit_transform(X_train, y_train)
# X_test = lda.transform(X_test)

# from sklearn.decomposition import PCA
# pca = PCA(n_components = None)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# explained_variance = pca.explained_variance_ratio_

# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
# import xgboost as xgb
# # -----------------------------------------------------------------------------
# # K-neighbourhood
# # -----------------------------------------------------------------------------

# alg_ngbh = KNeighborsClassifier(n_neighbors=3)
# scores = cross_val_score(alg_ngbh, X_train, y_train,cv = 10, n_jobs=-1)
# print("Accuracy (k-neighbors): {}/{}".format(scores.mean(), scores.std()))

# # -----------------------------------------------------------------------------
# # kernel SVM
# # -----------------------------------------------------------------------------

# alg_sgd = SVC(kernel='linear')
# scores = cross_val_score(alg_sgd, X_train, y_train, cv=10, n_jobs=-1)
# print("Accuracy (sgd): {}/{}".format(scores.mean(), scores.std()))

# # -----------------------------------------------------------------------------
# # svm
# # -----------------------------------------------------------------------------

# alg_svm = SVC(C=1.0)
# scores = cross_val_score(alg_svm, X_train, y_train, cv=10, n_jobs=-1)
# print("Accuracy (svm): {}/{}".format(scores.mean(), scores.std()))

# # -----------------------------------------------------------------------------
# # naive bayes
# # -----------------------------------------------------------------------------

# alg_nbs = GaussianNB()
# scores = cross_val_score(alg_nbs, X_train, y_train, cv=10, n_jobs=-1)
# print("Accuracy (naive bayes): {}/{}".format(scores.mean(), scores.std()))


# # -----------------------------------------------------------------------------
# # logistic regression
# # -----------------------------------------------------------------------------

# alg_log = LogisticRegression(random_state=1)
# scores = cross_val_score(alg_log, X_train, y_train, cv=10, n_jobs=-1,)
# print("Accuracy (logistic regression): {}/{}".format(scores.mean(), scores.std()))

# # -----------------------------------------------------------------------------
# # random forest simple
# # -----------------------------------------------------------------------------

# alg_frst = RandomForestClassifier(random_state=1, n_estimators=500, min_samples_split=8, min_samples_leaf=2)
# scores = cross_val_score(alg_frst, X_train, y_train, cv=10, n_jobs=-1)
# print("Accuracy (random forest): {}/{}".format(scores.mean(), scores.std()))

# # -----------------------------------------------------------------------------
# # random forest auto
# # -----------------------------------------------------------------------------

# alg_frst_model = RandomForestClassifier(random_state=1)
# alg_frst_params = [{
#     "n_estimators": [350, 400, 450],
#     "min_samples_split": [6, 8, 10],
#     "min_samples_leaf": [1, 2, 4]
# }]
# alg_frst_grid = GridSearchCV(alg_frst_model, alg_frst_params, cv=10, refit=True, verbose=1, n_jobs=-1)
# alg_frst_grid.fit(X_train, y_train)
# alg_frst_best = alg_frst_grid.best_estimator_
# print("Accuracy (random forest auto): {} with params {}"
#       .format(alg_frst_grid.best_score_, alg_frst_grid.best_params_))

# -----------------------------------------------------------------------------
# XBoost auto
# -----------------------------------------------------------------------------
# ##XGBOOST 84.28%
# ald_xgb_model = xgb.XGBClassifier()
# ald_xgb_params = [
#     {"n_estimators": [100],
#      "max_depth": [4],
#      "learning_rate": [0.01]}
# ]
# alg_xgb_grid = GridSearchCV(ald_xgb_model, ald_xgb_params, cv=10, refit=True, verbose=1, n_jobs=1)
# alg_xgb_grid.fit( X_train, y_train)
# alg_xgb_best = alg_xgb_grid.best_estimator_
# print("Accuracy (xgboost auto): {} with params {}"
#       .format(alg_xgb_grid.best_score_, alg_xgb_grid.best_params_))

#ANN-----------------------------------------------------------------------------------
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 14))
# Adding the second hidden layer
# relu rectifier function reccomended for hidden layer
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
#sigmoid for output(softmax for more then 1 output)
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
#adam(stochastic gradient), (categorical)binary_crossentropy (logistical loss function)
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# parameters = [{'optimizers' : ['rmsprop', 'adam'],
# 'inits' : ['glorot_uniform', 'normal', 'uniform'],
# 'epochs' : [50, 100, 200, 400],
# 'batches' : [5, 10, 20]}]

# grid_search = GridSearchCV(estimator = classifier,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10,
#                            n_jobs = -1)
# grid_search = grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_