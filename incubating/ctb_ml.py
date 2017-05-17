from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

def getBestModel(trainX, trainY):
    logisticRegression = LogisticRegression()
    logisticRegression.fit(trainX, trainY)
    logisticRegressionAccuracy = round(logisticRegression.score(trainX, trainY) * 100, 2)
    # print(logisticRegressionAccuracy)

    svc = SVC()
    svc.fit(trainX, trainY)
    svcAccuracy = round(svc.score(trainX, trainY) * 100, 2)
    # print(svcAccuracy)

    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(trainX, trainY)
    knnAccuracy = round(knn.score(trainX, trainY) * 100, 2)
    # print(knnAccuracy)

    gaussianNaiveBayes = GaussianNB()
    gaussianNaiveBayes.fit(trainX, trainY)
    gaussianNaiveBayesAccuracy = round(gaussianNaiveBayes.score(trainX, trainY) * 100, 2)
    # print(gaussianNaiveBayesAccuracy)

    perceptron = Perceptron()
    perceptron.fit(trainX, trainY)
    perceptronAccuracy = round(perceptron.score(trainX, trainY) * 100, 2)
    # print(perceptronAccuracy)

    linearSvc = LinearSVC()
    linearSvc.fit(trainX, trainY)
    linearSvcAccuracy = round(linearSvc.score(trainX, trainY) * 100, 2)
    # print(linearSvcAccuracy)

    sgd = SGDClassifier()
    sgd.fit(trainX, trainY)
    sgdAccuracy = round(sgd.score(trainX, trainY) * 100, 2)
    # print(sgdAccuracy)

    decisionTree = DecisionTreeClassifier()
    decisionTree.fit(trainX, trainY)
    decisionTreeAccuracy = round(decisionTree.score(trainX, trainY) * 100, 2)
    # print(decisionTreeAccuracy)

    randomForest = RandomForestClassifier()
    randomForest.fit(trainX, trainY)
    randomForestAccuracy = round(randomForest.score(trainX, trainY) * 100, 2)
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
