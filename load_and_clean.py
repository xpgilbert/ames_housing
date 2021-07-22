#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:09:33 2021

@author: Gilly
"""

# Load and Clean Data
## Ames, Iowa Housing Dataset

## Imports
## Linear Algebra, Data Science
import numpy as np
import pandas as pd
## Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
#%%
## Read and ready data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
#%%
## We know we have outliers from our exploratory analysis, lets remove these
## Overall quality outliers
# qual_out_sp = train[(train['SalePrice'] < 200000)
#                          & (train['OverallQual'] >= 9)]['SalePrice'].mean()
train = train.drop(train[(train['SalePrice'] < 200000)
                         & (train['OverallQual'] >= 9)].index)
## Above ground living area outliers
train = train.drop(train[(train['SalePrice'] < 200000)
                         & (train['GrLivArea'] > 4000)].index)
## Four car garage outliers
train = train.drop(train[(train['SalePrice'] < 300000)
                         & (train['GarageCars'] >= 3)].index)
## Large garage outliers
train = train.drop(train[(train['SalePrice'] < 300000)
                         & (train['GarageArea'] >= 1200)].index)
## Many rooms above ground outliers
train = train.drop(train[(train['SalePrice'] < 300000)
                         & (train['TotRmsAbvGrd'] > 12)].index)
## A missing value in these columns represents the None class
none_cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
       'MiscFeature']
cols = df_train.columns[~df_train.columns.isin(none_cols)]
print(len(cols))
print(len(none_cols))
#%%
df_train = train.drop('SalePrice', axis=1)
target = train['SalePrice']
#%%
## Get training data means and modes to impute
def get_imputes(data):
    imputes = {}
    cols = df_train.columns[~df_train.columns.isin(none_cols)]
    for col in cols:
        if data[col].dtype == 'object':
            imputes[col] = data[col].mode()[0]
        else:
            imputes[col] = data[col].mean()
    return imputes
## Impute missing values function
def impute_missing(data, imputes):
    data = data.copy()
    for acol in none_cols:
        data[acol] = data[acol].fillna('None')
    for bcol in imputes.keys():
        data[bcol] = data[bcol].fillna(imputes[bcol])
    return data
imputes = get_imputes(df_train)
df_train = impute_missing(df_train, imputes)
print(df_train.isnull().sum().max())
#%%
## Apply same imputes to test data using imputes from training set
## Check that the test set has similar distribution of nulls, where
## we had the same number of nulls for certain categories of features,
## such as the garage.
print(test.isnull().sum().sort_values(ascending=False)[:20])
#%%
def update_imputes(data, imputes):
    n_0 = 1460
    imputes = imputes.copy()
    for col in data.columns:
        if col not in imputes.keys():
            if data[col].dtype == 'object':
                imputes[col] = data[col].mode()[0]
            else:
                imputes[col] = data[col].mean()
        elif data[col].dtype != 'object':
            og = np.empty(n_0)
            og.fill(imputes[col])
            imputes[col] = np.mean(data[col].append(og))
    return imputes
df_test = impute_missing(test, imputes)
print(df_test.isnull().sum().sort_values(ascending=False)[:10])
#%%
