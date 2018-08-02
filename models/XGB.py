"""
Individual Machine Learning Model (SciKit-Learn)

Model: Xtreme Gradient Boosting
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import matplotlib.pyplot as plt

#Read CSV
training = pd.read_csv("../input/train.csv")
testing = pd.read_csv("../input/test.csv")

#Method to fill in NA values
def fillna(attribute, value):
    training[attribute] = training[attribute].fillna(value)
    testing[attribute] = testing[attribute].fillna(value)

#Filling in values
training['LotFrontage'] = training['LotFrontage'].fillna(training['LotFrontage'].mean())
testing['LotFrontage'] = testing['LotFrontage'].fillna(testing['LotFrontage'].mean())
fillna("Alley", "None")
fillna("MasVnrType", "None")
fillna("MasVnrArea", 0)
fillna("BsmtQual", "None")
fillna("BsmtCond", "None")
fillna("BsmtExposure", "None")
fillna("BsmtFinType1", "None")
fillna("BsmtFinType2", "None")
fillna("Electrical", "SBrkr")
fillna("FireplaceQu", "None")
fillna("GarageType", "None")
fillna("GarageYrBlt", 0)
fillna("GarageFinish", "None")
fillna("GarageQual", "None")
fillna("GarageCond", "None")
fillna("PoolQC", "None")
fillna("Fence", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
fillna("MiscFeature", "None")
for e in testing.columns:
    testing[e] = testing[e].fillna(testing[e].mode().iloc[0])


#Creating numpy arrays for training categorical data, numerical data, and result labels
train_cat = training.iloc[:, np.r_[1:3, 5, 7:9, 10:17, 21:25, 27:33, 35, 39:42, 53, 55, 57:58, 60, 63:65, 73:75, 76, 78:79] ]
train_num = training.iloc[:, np.r_[3:4, 17:20, 26, 34, 36:38, 43:52, 54, 56, 59, 61:62, 66:72, 75, 77] ]
train_result = training.iloc[:, 80].values.reshape(-1, 1)

#Creating numpy arrays for testing categorical and numerical data
test_cat = testing.iloc[:, np.r_[1:3, 5, 7:9, 10:17, 21:25, 27:33, 35, 39:42, 53, 55, 57:58, 60, 63:65, 73:75, 76, 78:79] ]
test_num = testing.iloc[:, np.r_[3:4, 17:20, 26, 34, 36:38, 43:52, 54, 56, 59, 61:62, 66:72, 75, 77] ]

#Aggregating categorical and numerical data from training and testing
cat = np.concatenate([train_cat, test_cat], axis=0)
num = np.concatenate([train_num, test_num], axis=0)

#Create Label Encoder and Onehot encoder
le = LabelEncoder()
enc = OneHotEncoder()

cat = pd.DataFrame(cat)

#One hot encoding of All Categories
overall_enc = cat.astype(str).apply(le.fit_transform)
enc.fit(overall_enc)
onehot = enc.transform(overall_enc).toarray()

#Split Categories into train and test
train_onehot = onehot[:1460]
test_onehot = onehot[1460:]

#Final Training and testing numpy arrays
train = np.concatenate([train_num, train_onehot], axis=1)
test = np.concatenate([test_num, test_onehot], axis=1)

#Validation split
validate = train[1200:]
train_sub = train[:1200]
validate_result = train_result[1200:]
train_sub_result = train_result[:1200]


#Initialize xgb model
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

n_folds = 5
#K-Fold Cross Validation for RMSLE error
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, train, train_result.reshape(-1), scoring="neg_mean_squared_log_error", cv = kf))
    return(rmse)

#Fit xgb model using training dataset
model_xgb.fit(train, train_result)

#print rmsle error
score = rmsle_cv(model_xgb)
print("Average Score: {:f} with std of {:f}".format(score.mean(), score.std()))

#final prediction
npresult = model_xgb.predict(test).reshape(-1)

#Export prediction as csv
nparange = np.arange(1461, 2920)
pd.DataFrame({'Id': nparange, 'SalePrice': npresult}).to_csv('XGB.csv', index =False)
