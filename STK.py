'''

Authors: Xiangyu Zhao
Date: Thursday, 9th December, 2021
Title: single task learning using Tensorflow for DL, sklearn for ML

'''

import tensorflow as tf
from os.path import abspath
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample
import numpy as np

# define constants
RANDOM_SEED = 29

data_path = abspath("./dataset/labels/extreme_hs_german_bert_features_binary_labels/acous_ling.csv")
data = pd.read_csv(data_path)
data_female = data[data['Gender'] == 'f']
data_male = data[data['Gender'] == 'm']

'''
# use data from target and partner
data_female = data_female.drop(['good_bad', 'stress_relaxed', 'peaceful_angry', 'happy_sad', 'Gender', 'Name'], axis=1)
data_combined = pd.merge(data_male, data_female, on=['ID'])
X = data_combined.drop(['ID', 'Gender', 'good_bad', 'stress_relaxed', 'peaceful_angry', 'happy_sad', 'Name'], axis=1)
Y = data_combined.loc[:, 'happy_sad'].to_numpy()
'''

# only use target subject data
X = data_male.drop(['ID', 'Gender', 'good_bad', 'stress_relaxed', 'peaceful_angry', 'happy_sad', 'Name'], axis=1)
Y = data_male.loc[:, 'happy_sad'].to_numpy()

'''
Y = tf.one_hot(Y, 2, dtype=tf.int64).numpy() #one hot encoding
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
X_train_upsample, Y_train_upsample = resample(X_train[Y_train == 1], Y_train[Y_train == 1],
                                              replace=True, n_samples=X_train[Y_train == 0].shape[0])
X_train_balanced = np.vstack((X_train[Y_train == 0], X_train_upsample))
Y_train_balanced = np.concatenate((Y_train[Y_train == 0], Y_train_upsample))
'''


def ml_classifier(x, y):
    # machine learning model

    # Support Vector Classifier
    svc_parameters = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
                      'gamma': ('scale', 'auto')}
    svc = svm.SVC(class_weight='balanced')

    # Decision Tree
    dt_parameters = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random'),
                     'max_features': ('auto', 'sqrt', 'log2')}
    dt = DecisionTreeClassifier(class_weight='balanced')

    # Random Forest
    rf_parameters = {'n_estimators': [10, 50, 100, 500, 1000], 'criterion': ('gini', 'entropy'),
                     'max_features': ('auto', 'sqrt', 'log2')}
    rf = RandomForestClassifier(class_weight='balanced')

    # AdaBoost Classifier
    ada_parameters = {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 1.0]}
    ada = AdaBoostClassifier()

    # Ridge Classifier
    rc_parameters = {'alpha': [0.00001, 0.0001, 0.001, 0.01]}
    rc = linear_model.RidgeClassifier(class_weight='balanced')

    # Grid Search for best parameter
    classifier = GridSearchCV(svc, svc_parameters, scoring='f1_macro')
    classifier.fit(X, Y)
    print('best params:')
    print(classifier.best_params_)
    best_model = classifier.best_estimator_

    # 10 cross validation for best parameter model
    scores = cross_val_score(best_model, X, Y, cv=5, scoring='f1_macro')
    print(str(scores.mean()) + '+-' + str(scores.std()))
    print('finish grid search and cross validation')
    return scores


def ml_regressor(x, y):
    svr_parameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'gamma': ('scale', 'auto'), 'C': [0.001, 0.01, 0.1, 1]}
    svr = SVR()

    regressor = GridSearchCV(svr, svr_parameters)
    regressor.fit(x, y)
    print('best params:')
    print(regressor.best_params_)
    best_model = regressor.best_estimator_

    return best_model


def dl_train(x, y):
    # deep learning model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=(1888,)))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    class_weight = {0: 1, 1: 10}
    model.fit(x_train, y_train, epochs=10, batch_size=8, class_weight=class_weight)
    model.evaluate(x_test, y_test)

    return model


ml_classifier(X, Y)
# dl_train(X, Y)

'''
# decision fusion for acoustic and linguistic regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=RANDOM_SEED)
X_train_acoustic = X_train.loc[:, 'F0semitoneFrom27.5Hz_sma3nz_amean-0':'equivalentSoundLevel_dBp-1']
X_train_linguistic = X_train.loc[:, 'Col0':'Col767']
X_test_acoustic = X_test.loc[:, 'F0semitoneFrom27.5Hz_sma3nz_amean-0':'equivalentSoundLevel_dBp-1']
X_test_linguistic = X_test.loc[:, 'Col0':'Col767']
acoustic_regressor = ml_regressor(X_train_acoustic, Y_train)
linguistic_regressor = ml_regressor(X_train_linguistic, Y_train)

acoustic_score = acoustic_regressor.predict(X_test_acoustic)
predict_acoustic = acoustic_score
predict_acoustic[predict_acoustic < 0.3] = 0
predict_acoustic[predict_acoustic >= 0.3] = 1
accuracy_acoustic = accuracy_score(Y_test, predict_acoustic)
print('acoustic accuracy:' + str(accuracy_acoustic))

linguistic_score = linguistic_regressor.predict(X_test_linguistic)
predict_linguistic = linguistic_score
predict_linguistic[predict_linguistic < 0.3] = 0
predict_linguistic[predict_linguistic >= 0.3] = 1
accuracy_linguistic = accuracy_score(Y_test, predict_linguistic)
print('linguistic accuracy:' + str(accuracy_linguistic))

average_score = np.mean([acoustic_score, linguistic_score], axis=0)
predict = average_score
predict[predict < 0.3] = 0
predict[predict >= 0.3] = 1
accuracy = accuracy_score(Y_test, predict)
print('fusion accuracy:' + str(accuracy))
'''

'''
# test
Y_pred = classifier.predict(X_test)
# Y_pred[Y_pred < 0.5] = 0
# Y_pred[Y_pred >= 0.5] = 1
accuracy = accuracy_score(Y_test, Y_pred)
f1_micro = f1_score(Y_test, Y_pred, average='micro')
f1_macro = f1_score(Y_test, Y_pred, average='macro')
print("accuracy is " + str(accuracy))
print("f1 score micro is " + str(f1_micro))
print("f1 score macro is " + str(f1_macro))
'''

print('finish')
