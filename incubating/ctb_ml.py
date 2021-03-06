from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression

import pandas as pd

import numpy as np

def getBestClassificationModel(inputX, inputY):
    trainX, validationX, trainY, validationY = train_test_split(inputX, inputY, random_state=1)

    logisticRegression = LogisticRegression()
    logisticRegression.fit(trainX, trainY)
    logisticRegressionAccuracy = round(logisticRegression.score(validationX, validationY) * 100, 2)
    # print(logisticRegressionAccuracy)

    svc = SVC()
    svc.fit(trainX, trainY)
    svcAccuracy = round(svc.score(validationX, validationY) * 100, 2)
    # print(svcAccuracy)

    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(trainX, trainY)
    knnAccuracy = round(knn.score(validationX, validationY) * 100, 2)
    # print(knnAccuracy)

    gaussianNaiveBayes = GaussianNB()
    gaussianNaiveBayes.fit(trainX, trainY)
    gaussianNaiveBayesAccuracy = round(gaussianNaiveBayes.score(validationX, validationY) * 100, 2)
    # print(gaussianNaiveBayesAccuracy)

    perceptron = Perceptron()
    perceptron.fit(trainX, trainY)
    perceptronAccuracy = round(perceptron.score(validationX, validationY) * 100, 2)
    # print(perceptronAccuracy)

    linearSvc = LinearSVC()
    linearSvc.fit(trainX, trainY)
    linearSvcAccuracy = round(linearSvc.score(validationX, validationY) * 100, 2)
    # print(linearSvcAccuracy)

    sgd = SGDClassifier()
    sgd.fit(trainX, trainY)
    sgdAccuracy = round(sgd.score(validationX, validationY) * 100, 2)
    # print(sgdAccuracy)

    decisionTree = DecisionTreeClassifier()
    decisionTree.fit(trainX, trainY)
    decisionTreeAccuracy = round(decisionTree.score(validationX, validationY) * 100, 2)
    # print(decisionTreeAccuracy)

    randomForest = RandomForestClassifier()
    randomForest.fit(trainX, trainY)
    randomForestAccuracy = round(randomForest.score(validationX, validationY) * 100, 2)
    # print(randomForestAccuracy)

    models = pd.DataFrame({
        'Model': [
            logisticRegression,
            svc,
            knn,
            gaussianNaiveBayes,
            perceptron,
            linearSvc,
            sgd,
            decisionTree,
            randomForest
        ],
        'Accuracy': [
            logisticRegressionAccuracy,
            svcAccuracy,
            knnAccuracy,
            gaussianNaiveBayesAccuracy,
            perceptronAccuracy,
            linearSvcAccuracy,
            sgdAccuracy,
            decisionTreeAccuracy,
            randomForestAccuracy
        ]
    })
    models.sort_values(by='Accuracy', ascending=False, inplace=True)
    print(models)

    return models['Model'].iloc[0]

def getBestRegressionModel(inputX, inputY):
    trainX, validationX, trainY, validationY = train_test_split(inputX, inputY, random_state=1)

    linearRegression = LinearRegression()
    linearRegression.fit(trainX, trainY)
    linearRegressionMse = np.mean((linearRegression.predict(validationX) - validationY) ** 2)
    print(linearRegressionMse)
    linearRegressionScore = linearRegression.score(validationX, validationY)
    print(linearRegressionScore)

    models = pd.DataFrame({
        'Model': [
            linearRegression
        ],
        'Score': [
            linearRegressionScore
        ]
    })
    models.sort_values(by='Score', ascending=False, inplace=True)
    print(models)

    return models['Model'].iloc[0]
