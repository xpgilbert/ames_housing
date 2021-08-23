#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 23:03:19 2021

@author: xpgilbert

Modeling with Python
Ames, Iowa Housing Dataset
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
from sklearn.preprocessing import StandardScaler
## Custome PreProcessing
from functions import Process
## Train-test-split, KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
## Metrics
from sklearn.metrics import mean_squared_error
## XGBoost
import xgboost as xgb
## OLS
import statsmodels.api as sm

## Set constants
random_state = 42
#%%
## Import csv to pandas DataFrames
df_train = pd.read_csv('data/df_train.csv')
df_test = pd.read_csv('data/df_test.csv')
target = pd.read_csv('data/target.csv').values.ravel()
## Remove skewness, as done from the exploratory
target = np.log1p(target)
## Extract Test ID's for submission
test_ids = df_test['Id']
#%%
'''
Here, we will first mean encode for some variables. The categorical
variables will be treated individually and then we will bin some numerical
variables into some categorical.
'''
processor = Process()
## Add target variable for training
df_train['target'] = target
## Create bin dictionary for binning numeric variables
# bins = {}
# bins['YearBuilt'] = [0, 1970, 1990, 2010, 2030]
# # #bins['']
# ## Bin numeric variables into categorical
# df_train = processor.bin_numerics(df_train, bins)
# df_test = processor.bin_numerics(df_test, bins)
#%%
## Engineer new variables
df_train['years_to_remod'] = df_train['YearRemodAdd'] -  df_train['YearBuilt']
df_test['years_to_remod'] = df_test['YearRemodAdd'] -  df_test['YearBuilt']
#%%
## Total living space including finished basement
basements = ['GLQ', 'ALQ', 'BLQ', 'Rec']
def total_space(data, basements):
    for val in data['BsmtFinType1']:
        if val in basements:
            fin1 = data['BsmtFinSF1']
        else:
            fin1 = 0
    for val in data['BsmtFinType2']:
        if val in basements:
            fin2 = data['BsmtFinSF2']
        else:
            fin2 = 0
    totals = data['GrLivArea'] + fin1 + fin2
    return totals
df_train['total_space'] = total_space(df_train, basements)
df_test['total_space'] = total_space(df_test, basements)
#%%
## Mean Encode some variables
cols = ['Neighborhood', 'MSSubClass', 'Functional', 'MiscFeature'
        , 'BsmtFinType1', 'BsmtFinType2']
for col in cols:
    df_train = processor.mean_encode_train(df_train, col)
for col in cols:
    df_test = processor.mean_encode_new(df_test, col)
df_train = df_train.drop('target', axis=1)
#%%
## Convert categorical to dummies
df_train = processor.one_hot_encode(df_train)
df_test = processor.one_hot_encode(df_test)
#%%
'''
First we will want to reduce the dimensionality of our training set.  We have
over 200 features, the vast majority of which are binary dummy features.  To
start, we can use VarianceThreshold to remove features with low variance.
'''
# ## VarianceThreshold
# ## For binary, Var[x] = p(1-p)
selector = VarianceThreshold(threshold=.8*(1-.8)) ## set threshold to 80%
# ## Only interested in binary variables
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
selector = SelectKBest(f_regression, k=30)
selector.fit(df_train, target)
temp_df = df_train[df_train.columns[selector.get_support()]]
numerics = temp_df.columns
selected = selected.append(temp_df.columns)
## Inlcude all dwelling types for modeling
# selected = selected.append(df.filter(regex='MSSubClass.*').columns)
# selected = list(set(selected))
#%%
## Reduce dimensions
df_train = processor.select(df, selected)
df_test = processor.select(df_test, selected)
#%%
## Match columns
for col in df_train.columns:
    if col not in df_test.columns:
        df_test[col] = 0
df_test = df_test.reindex(df_train.columns, axis=1)
#%%
## Pairplot of the 15 best variables to see their distributions.
# pair = sns.pairplot(temp_df)
# plt.title('Distributions of 15 best numeric variables')
# plt.show()
# pair.savefig('plots/pairplot.png')

#%%
'''
Looks like most of the variables are normally distributed but some have skewed
distributions.  We will normalize these features with a MinMaxScaler since
they have lots of zeroes.
'''
scaler = StandardScaler()
df_train[numerics] = scaler.fit_transform(df_train[numerics])
df_test[numerics] = scaler.fit_transform(df_test[numerics])
# df_train = scaler.fit_transform(df_train)
# df_test = scaler.fit_transform(df_test)
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
'''
So definitely need to tweak some hyperparameters.  We can also try using
a different model but for now lets use grid search for the xgboost.
'''
## Create dmatrices
dmat_train = xgb.DMatrix(X_train, label=y_train)
dmat_test = xgb.DMatrix(X_test, label=y_test)

xgbr = xgb.XGBRFRegressor()
params = {
    'objective' : ['reg:squarederror'],
    'max_depth' : [40,60],
    'eta' : [0.03],
    'min_child_weight' : [2,4],
    'n_estimators' : [600,800]
    }
xgbr_grid = GridSearchCV(xgbr, params, cv=5, n_jobs=-1, verbose=2)
xgbr_grid.fit(X_train, y_train)
#%%
## Pull best parameters for new model
best_params = xgbr_grid.best_params_
model = xgb.train(best_params,
                  dmat_train,
                  num_boost_round = 500,
                  early_stopping_rounds=20,
                  evals=[(dmat_test, 'Test')]
              )
#%%
dxtest = xgb.DMatrix(X_test)
xgb_pred = model.predict(dxtest)
mse = mean_squared_error(y_test, xgb_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))
#%%
## Modeling using OLS
ols = sm.OLS(y_train, X_train)
results = ols.fit()
ols_pred = results.predict(X_test)
mse = mean_squared_error(y_test, ols_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))
#%%
## Generate predictions based on test set with XGBoost
dtest = xgb.DMatrix(df_test)
predictions = model.predict(dtest)
## Return to non-normalized scale
predictions = np.expm1(predictions)
## Create submission dataframe
submission = pd.DataFrame.from_dict({'Id':test_ids, 'SalePrice':predictions})
submission.to_csv('submission.csv', index=False)
#%%
print(np.expm1(mse))
print(np.expm1(mse**(1/2.0)))
