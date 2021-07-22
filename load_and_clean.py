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

class Load():
    def __init__(self, data, none_cols):
        '''
        Parameters
        ----------
        data : pandas DataFrame
            dataframe on which to train the model later.
        none_cols : list
            strings of column names in data where a missing value
            represents the class None.
        '''
        self.data = data
        self.none_cols = none_cols
        self.get_imputes()
    ## Get imputes from training data to use with future data
    def get_imputes(self):
        '''
        Generates a dictionary of the means of numerical values and
        modes of categorical data from the training set used to
        initialize this class.
        '''
        data = self.data
        none_cols = self.none_cols
        imputes = {}
        cols = data.columns[~data.columns.isin(none_cols)]
        for col in cols:
            if data[col].dtype == 'object':
                imputes[col] = data[col].mode()[0]
            else:
                imputes[col] = data[col].mean()
        self.imputes = imputes
    ## Update imputes with new data
    def update_imputes(self, new_data):
        '''
        Parameters
        ----------
        new_data : pandas DataFrame
            Previously unseen data with which we can update our
            imputes dictionary.  This is used to impute missing values
            in this new dataset.
        '''
        data = new_data
        imputes = self.imputes
        n_0 = self.data.shape[0]
        for col in data.columns:
            if col not in imputes.keys():
                if data[col].dtype == 'object':
                    imputes[col] = data[col].mode()[0]
                else:
                    imputes[col] = data[col].mean()
            elif data[col].dtype != 'object':
                og = np.empty(n_0)
                og.fill(imputes[col])
                imputes[col] = np.mean(data[col].append(pd.Series(og)))
        self.imputes = imputes
    ## Impute missing values function
    def impute_missing(self, data):
        '''
        Parameters
        ----------
        data : pandas DataFrame
            dataframe on which to impute values from this class's
            imputes dictionary object.

        Returns
        -------
        data : pandas DataFrame
            dataframe imputed with values from this class's 
            imputes dictionary object.  If this is not the data used
            to initialize this class, the method update_imputes should
            be called before this method.
        '''
        imputes = self.imputes
        none_cols = self.none_cols
        for acol in none_cols:
            data[acol] = data[acol].fillna('None')
        for bcol in imputes.keys():
            data[bcol] = data[bcol].fillna(imputes[bcol])
        return data
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
print(df_train['PoolQC'].unique())
