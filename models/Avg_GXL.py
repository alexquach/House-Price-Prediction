"""
Averaging Model

Base Models: Gradient Boosting, Xtreme Gradient Boosting, Light Gradient Boosting
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



#Initializing individual machine learning models
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

n_folds = 5
#K-Fold Cross Validation for RMSLE error
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, train, train_result.reshape(-1), scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

#Averaging model
class AverageModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, x, y):
        self.models_ = [clone(z) for z in self.models]

        for model in self.models_:
            model.fit(x, y)

        return self

    def predict(self, x):
        predictions = np.column_stack([model.predict(x) for model in self.models_])
        return np.mean(predictions, axis = 1)

#Constructing average model with individual models
average_model = AverageModel(models = (GBoost, model_xgb, model_lgb))

#Fitting the model
average_model.fit(train, train_result)

#Resulting error
score = rmsle_cv(average_model)
print("Average Score: {:f} with std of {:f}".format(score.mean(), score.std()))

#Final prediction
npresult = average_model.predict(test)

#Export final prediction as csv
nparange = np.arange(1461, 2920)
pd.DataFrame({'Id': nparange, 'SalePrice': npresult}).to_csv('Avg_GXL.csv', index =False)
