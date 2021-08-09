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
## Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
## Feature Selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

#%%
## Import csv to pandas DataFrames
df_train = pd.read_csv('data/df_train.csv')
df_test = pd.read_csv('data/df_test.csv')
target = pd.read_csv('data/target.csv').values.ravel()
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
df = df_train
binary_columns = [col for col in df.columns if df[col].nunique() == 2]
binary = df_train[binary_columns]
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
df_train = df_train[df_train.columns[selector.get_support()]]
selected = selected.append(df_train.columns)

#%%
## Pairplot of the 15 best variables to see their distributions.
pair = sns.pairplot(df_train)
plt.title('Distributions of 15 best numeric variables')
plt.show()
pair.savefig('plots/pairplot.png')
#%%
'''
Looks like most of the variables are normally distribute but some have skewed
distributions.  We will normalize these features with a MinMaxScaler since
they have lots of zeroes.
'''