import pandas as pd

train = pd.read_csv('test/kaggle/house_prices/train.csv')
# print(train)

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data)

train = train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
# train = train.drop(train.loc[train['Electrical'].isnull()].index) # Makes worse.
# print(train.isnull().sum().max())

trainX = train.drop([
    "Id",
    # "MSSubClass",
    "MSZoning",
    # "LotFrontage", # Filtered when processing missing data.
    # "LotArea",
    "Street",
    # "Alley", # Filtered when processing missing data.
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    # "OverallQual",
    # "OverallCond",
    # "YearBuilt",
    # "YearRemodAdd",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    # "MasVnrType", # Filtered when processing missing data.
    # "MasVnrArea", # Filtered when processing missing data.
    "ExterQual",
    "ExterCond",
    "Foundation",
    # "BsmtQual", # Filtered when processing missing data.
    # "BsmtCond", # Filtered when processing missing data.
    # "BsmtExposure", # Filtered when processing missing data.
    # "BsmtFinType1", # Filtered when processing missing data.
    "BsmtFinSF1",
    # "BsmtFinType2", # Filtered when processing missing data.
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    # "1stFlrSF",
    # "2ndFlrSF",
    # "LowQualFinSF",
    # "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    # "FullBath",
    # "HalfBath",
    # "BedroomAbvGr",
    # "KitchenAbvGr",
    "KitchenQual",
    # "TotRmsAbvGrd",
    "Functional",
    # "Fireplaces",
    # "FireplaceQu", # Filtered when processing missing data.
    # "GarageType", # Filtered when processing missing data.
    # "GarageYrBlt", # Filtered when processing missing data.
    # "GarageFinish", # Filtered when processing missing data.
    "GarageCars",
    "GarageArea",
    # "GarageQual", # Filtered when processing missing data.
    # "GarageCond", # Filtered when processing missing data.
    "PavedDrive",
    # "WoodDeckSF",
    # "OpenPorchSF",
    # "EnclosedPorch",
    # "3SsnPorch",
    # "ScreenPorch",
    # "PoolArea",
    # "PoolQC", # Filtered when processing missing data.
    # "Fence", # Filtered when processing missing data.
    # "MiscFeature", # Filtered when processing missing data.
    # "MiscVal",
    "MoSold", # Doesn't look good for a feature.
    "YrSold", # Doesn't look good for a feature.
    "SaleType",
    "SaleCondition",
    "SalePrice"
], axis=1)
# print(trainX)

trainY = train["SalePrice"]
# print(trainY)

import ctb_ml

selectedModel = ctb_ml.getBestRegressionModel(trainX, trainY)
print(selectedModel)

test = pd.read_csv('test/kaggle/house_prices/test.csv')
# print(test)

testX = test.drop([
    "Id",
    # "MSSubClass",
    "MSZoning",
    "LotFrontage",
    # "LotArea",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    # "OverallQual",
    # "OverallCond",
    # "YearBuilt",
    # "YearRemodAdd",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "MasVnrArea",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinSF1",
    "BsmtFinType2",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    # "1stFlrSF",
    # "2ndFlrSF",
    # "LowQualFinSF",
    # "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    # "FullBath",
    # "HalfBath",
    # "BedroomAbvGr",
    # "KitchenAbvGr",
    "KitchenQual",
    # "TotRmsAbvGrd",
    "Functional",
    # "Fireplaces",
    "FireplaceQu",
    "GarageType",
    "GarageYrBlt",
    "GarageFinish",
    "GarageCars",
    "GarageArea",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    # "WoodDeckSF",
    # "OpenPorchSF",
    # "EnclosedPorch",
    # "3SsnPorch",
    # "ScreenPorch",
    # "PoolArea",
    "PoolQC",
    "Fence",
    "MiscFeature",
    # "MiscVal",
    "MoSold", # Doesn't look good for a feature.
    "YrSold", # Doesn't look good for a feature.
    "SaleType",
    "SaleCondition"
], axis=1)
# print(testX)

testY = selectedModel.predict(testX)

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": testY
})

submission.loc[submission["SalePrice"] < 0, "SalePrice"] = -submission["SalePrice"]
submission.to_csv("test/kaggle/house_prices/submission.csv", index=False)
