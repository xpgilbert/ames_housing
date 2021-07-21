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
##Use whitegrid in seaborn
sns.set_style('whitegrid')
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
fig.savefig('plots/numerical_correlations.png')
##Target Distribution
dist = sns.displot(target, kde=True)
plt.title('Target Distribution')
plt.show()
dist.savefig('plots/target_distribution.png')
#%%
##Target is not normally distributed, lets try taking the log
log = sns.displot(np.log1p(target), color='g', kde=True)
plt.title('Log(Target+1)')
plt.show()
log.savefig('plots/log_target+1.png')
#%%
'''
From the plots and information above, we can see that there are quite a lot
variables and that the target is correlated most with features relating to 
area and quality.  Also interesting is that the variables associated with
luxury features, such as a fireplace and wood deck, are highly correlated.
We also know to engineer the target variable such that it is normally
distributed.
'''
## Correlations just with SalePrice
print(train.corr()['SalePrice'].sort_values(ascending=False))
#%%
## Distribution of top 3 numerical features
o_qual = sns.boxplot(x='OverallQual', y='SalePrice', data=train)
plt.title('Overall Quality vs. Sale Price')
plt.show()
gr_area = sns.jointplot(x='SalePrice', y='GrLivArea', data=train)
gr_area.fig.suptitle('Above ground living area\n vs.SalePrice')
gr_area.fig.tight_layout()
gr_area.fig.subplots_adjust(top=0.9)
plt.show()
gar_n = sns.boxplot(x='GarageCars',y='SalePrice', data=train)
plt.suptitle('Number of Cars in Garage')
plt.title('vs. Sale Price')
plt.show()
o_qual.savefig('plots/overall_qual.png')
gr_area.savefig('plots/above_ground.png')
gar_n.savefig('plots/garace_cars.png')
#%%
''' This is a popular dataset so some of the feature engineering and 
exploratory analysis are either verbatim or inspired by notebooks Ive browsed
before starting my own analysis.  I will try to be explicit as possible when 
using strategies others have used, but may not link a source as some of these
methods are very common when training a model using this data set.
For example, the plots above show that we have some outliers.  Removing these
outliers is very common across the board. They are:
    * A property with quality = 10 but very low sale price
    * Properties with large living areas but very low sale price
    * Homes with 4 car garages sell lower than those with 3 car garages
We will remove these outliers here.
'''

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
#%%








