#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 18:40:08 2021

@author: Gilly
"""
# Exploratory Data Analysis
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
print(train.head())
df_train = train.drop('SalePrice', axis=1)
target = train['SalePrice']
#%%
## Show shapes and sizes
print(f'Train set has {df_train.shape[0]} observations and {df_train.shape[1]} features')
print(f'Test set has {test.shape[0]} observations and {test.shape[1]} features')
print(f'Target array has length {len(target)}')
#%%
##Information and Descriptions
print('df_train Info:')
df_train.info()
print('\n\n')
print('Columns with null values:')
print('Feature        Count')
print('--------------------')
print(df_train.isnull().sum()[df_train.isnull().sum()>0].sort_values(ascending=False))
#%%
##Correlation Heatmap
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(train.corr(), cbar=True, cmap='viridis', ax=ax)
plt.title('Correlation heatmap of numeric variables')
plt.show()
##Target Distribution
sns.displot(target, kde=True)
plt.title('Target Distribution')
plt.show()
#%%
##Target is not normally distributed, lets try taking the log
sns.displot(np.log1p(target), color='g', kde=True)
plt.title('Log(Target+1)')
plt.show()
#%%
'''
From the plots and information above, we can see that there are quite a lot
variables and that the target is correlated most with features relating to 
area and quality.  Also interesting is that the variables associated with
luxury features, such as a fireplace and wood deck, are highly correlated.
We also know to engineer the target variable such that it is normally
distributed.
'''
##Correlations just with SalePrice
print(train.corr()['SalePrice'].sort_values(ascending=False))
#%%
'''
As suspected, the variables most correlated are those that deal with quality
and physical area.
'''
## Get the list of numeric and categorical columns\n",
df = df_train
binary_columns = [col for col in df.columns if df[col].nunique() == 2]
print("Binary Columns : ", binary_columns)
categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
print("Categorical Columns : ", categorical_columns)
categorical_columns = binary_columns + categorical_columns
categorical_columns = list(set(categorical_columns))
numerical_columns = [col for col in df.columns if col not in categorical_columns]
print("Numerical Columns : ", numerical_columns)
#%%
## Get distributions of categorical variables
print('Category    Uniques')
print('-------------------')
for cat in categorical_columns:
    n = str(len(df[cat].unique()))
    print(f'{cat:16}{n:1}')