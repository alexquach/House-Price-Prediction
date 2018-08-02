"""
Meta Model

Base Models: Kernel Ridge Regression, Random Forest, ElasticNet,
             Gradient Boosting, Xtreme Gradient Boosting, Light Gradient Boosting
Meta Model: Lasso
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


#Initialize all individual machine learning models
lasso = make_pipeline(RobustScaler(), Lasso(alpha =1, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
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
forest_model = RandomForestRegressor()


n_folds = 5
#K-Fold Cross Validation for RMSLE error
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, train, train_result.reshape(-1), scoring="neg_mean_squared_log_error", cv = kf))
    return(rmse)

"""
Meta Model
    Several Base models
    One Meta model
"""
class MetaModel(BaseEstimator, RegressorMixin, TransformerMixin):
    #initialize base and meta models
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    #fitting overal model
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=156)

        #placeholder
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        #iterate through all models and training folds
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                #create clone of model
                instance = clone(model)
                self.base_models_[i].append(instance)

                #fit instance model with train index
                instance.fit(X[train_index], y[train_index])

                #prediction from instance model
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # train meta model using out-of-fold predictions
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    #create meta-features from averaged prediction of instances of base modelsv
    #then do final prediction by meta model on meta-features
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

#Construction of meta model
meta_model = MetaModel(base_models = (KRR, forest_model, ENet, GBoost, model_xgb, model_lgb), meta_model = lasso)

#Fitting meta model with training dataset
meta_model.fit(train, train_result.reshape(-1))

#print rmsle error
score = rmsle_cv(meta_model)
print("Average Score: {:f} with std of {:f}".format(score.mean(), score.std()))

#final prediction
npresult = meta_model.predict(test).reshape(-1)

nparange = np.arange(1461, 2920)
pd.DataFrame({'Id': nparange, 'SalePrice': npresult}).to_csv('MM_ALL.csv', index =False)
