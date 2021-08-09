#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 23:03:19 2021

@author: Gilly

Modeling with Python
Ames, Iowa Real Estate Dataset
"""

## Imports
## Linear Algebra, Data Science
import pandas as pd
import numpy as np
## Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
## Feature Selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
## Scaler
from sklearn.preprocessing import MinMaxScaler
## Custome PreProcessing
from functions import Process
## Train-test-split, KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
## Metrics
from sklearn.metrics import mean_squared_error
## XGBoost
import xgboost as xgb

## Set constants
random_state = 42
#%%
## Import csv to pandas DataFrames
df_train = pd.read_csv('data/df_train.csv')
df_test = pd.read_csv('data/df_test.csv')
target = pd.read_csv('data/target.csv').values.ravel()
## Remove skewness like from the exploratory
target = np.log1p(target)
## Extract Test ID's for submission
test_ids = df_test['Id']
#%%
'''
First we will want to reduce the dimensionality of our training set.  We have
over 200 features, the vast majority of which are binary dummy features.  To
start, we can use VarianceThreshold to remove features with low variance.
'''
## VarianceThreshold
## For binary, Var[x] = p(1-p) 
selector = VarianceThreshold(threshold=.85*(1-.85)) ## set threshold to 85%
## Only interested in binary variables
df = df_train.copy()
binary_columns = [col for col in df.columns if df[col].nunique() == 2]
binary = df[binary_columns]
#%%
## Remove binary
df_train = df_train.drop(binary_columns, axis=1)
## Fit selector
selector.fit(binary)
binary = binary[binary.columns[selector.get_support()]]
## Create list of binary features to select from df_test
selected = binary.columns
#%%
## For non-binary variables, we can use SelectKBest from Sklearn
selector = SelectKBest(f_regression, k=15)
selector.fit(df_train, target)
temp_df = df_train[df_train.columns[selector.get_support()]]
numerics = temp_df.columns
selected = selected.append(temp_df.columns)
## Inlcude all dwelling types for modeling
selected = selected.append(df.filter(regex='MSSubClass.*').columns)
selected = list(set(selected))
#%%
## Reduce dimensions
processor = Process()
df_train = processor.select(df, selected)
df_test = processor.select(df_test, selected)
#%%
## Match columns
for col in df_train.columns:
    if col not in df_test.columns:
        df_test[col] = 0
#%%
## Pairplot of the 15 best variables to see their distributions.
# pair = sns.pairplot(temp_df)
# plt.title('Distributions of 15 best numeric variables')
# plt.show()
# pair.savefig('plots/pairplot.png')
#%%
'''
Looks like most of the variables are normally distribute but some have skewed
distributions.  We will normalize these features with a MinMaxScaler since
they have lots of zeroes.
'''
scaler = MinMaxScaler()
df_train[numerics] = scaler.fit_transform(df_train[numerics])
df_test[numerics] = scaler.fit_transform(df_test[numerics])
#%%
## Train test spliit
X_train, X_test, y_train, y_test = train_test_split(df_train, target,
                                                    test_size=0.2,
                                                    random_state=random_state)
## Initiate XGBoost Regressor
xgbr = xgb.XGBRegressor()
## Fit to training set
xgbr.fit(X_train, y_train)
kfold = KFold(n_splits=5, shuffle=True)
kf_cv_scores = cross_val_score(xgbr, X_train, y_train, cv=kfold)
print(f'K-fold CV average score:  {kf_cv_scores.mean()}')
#%%
'''
Looks like our model is just about 88% accurate with the training set.  As a
basic start, looks good.  Lets generate predictions on our validation set.
'''
## Fit model
## Generate predictions
y_pred = xgbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))
#%%

predictions = xgbr.predict(df_test)
submission = pd.DataFrame.from_dict({'Id':test_ids, 'SalePrice':predictions})
print(submission)






