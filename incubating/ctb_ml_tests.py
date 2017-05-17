import pandas as pd

train = pd.read_csv('train.csv')
# print(train)

median_age = train.Age.dropna().median()
# print(median_age)

median_fare = train.Fare.dropna().median()
# print(median_fare)

freq_port = train.Embarked.mode()[0]
# print(freq_port)

def quantify(dataSet):
    dataSet['Sex'] = dataSet['Sex'].map({
        'female': 1,
        'male': 0
    }).astype(int)

    dataSet['Age'] = dataSet['Age'].fillna(median_age)
    dataSet['AgeBand'] = pd.cut(train['Age'], 5, labels=[4, 3, 2, 1, 0]).astype(int)

    dataSet['Fare'] = dataSet['Fare'].fillna(median_fare)
    dataSet['FareBand'] = pd.qcut(train['Fare'], 3, labels=[0, 1, 2]).astype(int)

    dataSet['Embarked'] = dataSet['Embarked'].fillna(freq_port)
    dataSet['Embarked'] = dataSet['Embarked'].map({
        'C': 2,
        'Q': 1,
        'S': 0
    }).astype(int)

quantify(train)

trainX = train.drop([
    "PassengerId", "Survived", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Age", "Fare", "FareBand"
], axis=1)
# print(trainX)

trainY = train["Survived"]
# print(trainY)

import ctb_ml

selectedModel = ctb_ml.getBestModel(trainX, trainY)
print(selectedModel)

test = pd.read_csv('test.csv')
# print(test)

quantify(test)

testX = test.drop([
    "PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Age", "Fare", "FareBand"
], axis=1)
# print(testX)

testY = selectedModel.predict(testX)
# print(testY)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": testY
})
submission.to_csv("submission.csv", index=False)
