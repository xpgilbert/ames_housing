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
from functions import Load
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
#%%
df_train = train.drop('SalePrice', axis=1)
target = train['SalePrice']
#%%
## Check target series does not have missing values
assert target.isnull().sum() == 0, "Target has missing variables"
#%%
## A missing value in these columns represents the None class
none_cols = ['Alley','MasVnrType','BsmtQual','BsmtCond',
                 'BsmtExposure','BsmtFinType1','BsmtFinType2',
                 'FireplaceQu','GarageType','GarageFinish','GarageQual',
                 'GarageCond','PoolQC','Fence','MiscFeature']
loader = Load(df_train, none_cols)
df_train = loader.impute_missing(df_train)
#%%
## Apply same imputes to test data using imputes from training set
## Check that the test set has similar distribution of nulls, where
## we had the same number of nulls for certain categories of features,
## such as the garage.
df_test = test
print(df_test.isnull().sum().sort_values(ascending=False)[:20])
#%%
loader.update_imputes(df_test)
df_test = loader.impute_missing(df_test)
print(df_test.isnull().sum().sort_values(ascending=False)[:10])
#%%

