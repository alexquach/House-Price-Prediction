"""
Meta Model

Base Models: Lasso, ElasticNet, GradientBoosting
Meta Model : 0-Hidden layer Neural Network
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



#Initialize individual Machine Learning models
lasso = make_pipeline(RobustScaler(), Lasso(alpha =1, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

#Fit ML models on training data
lasso.fit(train, train_result)
ENet.fit(train, train_result)
GBoost.fit(train, train_result)

#Aggregate ML model predictions
meta_train = np.vstack([lasso.predict(train), ENet.predict(train), GBoost.predict(train)]).T
meta_validate = meta_train[1200:]
meta_train_sub = meta_train[:1200]
meta_test = np.vstack([lasso.predict(test), ENet.predict(test), GBoost.predict(test)]).T

#Method to retrieve random batches from aggregated ML model predictions
def meta_random_batch(batch_size):
    rand = random.randint(0, len(meta_train_sub)-1)
    t = meta_train_sub[rand]
    r = train_result[rand]
    for _ in range(batch_size-1):
        rand = random.randint(0, len(meta_train_sub)-1)
        t = np.vstack([t, meta_train_sub[rand]])
        r = np.vstack([r, train_result[rand]])
    return t, r

#Input layer
x = tf.placeholder(tf.float32, [None, meta_train.shape[1]])

#weights and bias
w = tf.Variable( tf.random_normal([meta_train.shape[1], 1]) )
b = tf.Variable( tf.random_normal([1]))
#prediction
y = tf.matmul(x, w) + b

#ground truth
y_ = tf.placeholder(tf.float32, [None, 1])

#create session and intialize variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#msle loss
overall_loss = tf.reduce_mean( tf.square( tf.log(y) - tf.log(y_) ) )

#training using Adam Optimizer
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(overall_loss)


#Iterate through timesteps
for _ in range(100000):
    #Retrieve random batch
    batch_xs, batch_ys = meta_random_batch(100)

    #train on random batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    #Reset weights if msle_loss is nan, or many weights are nan
    a = sess.run(overall_loss, feed_dict={x: batch_xs, y_: batch_ys})
    if(np.isnan(w.eval().reshape(-1)).sum() > 100 or a!=a):
        print("Reset weights")
        w = tf.Variable( tf.random_normal( [meta_train.shape[1], 1] ) )
        tf.global_variables_initializer().run()

    #Status markers
    if( _ % 1000 == 0):
        print("Epoch" + str(_) )
        print(sess.run(overall_loss, feed_dict={x: meta_validate, y_: validate_result}))

    #Final Predictions
    result = sess.run(y, feed_dict={x: meta_test} )
    result = result.reshape(-1)

sess.close()

#Export testing predictions as csv
nparange = np.arange(1461, 2920)
pd.DataFrame({'Id': nparange, 'SalePrice': result}).to_csv('MM_AEG_NN.csv', index =False)
