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
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
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
## Correlation heatmap of top 10 features
cols = train.corr().sort_values(by='SalePrice', ascending=False)[:10].index
corr = train[cols].corr()
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(corr, cmap = 'coolwarm', cbar=True, annot=True)
plt.title('Correlations of top 10 variables')
plt.show()
plt.savefig('plots/top10_corr.png')
#%%
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(train.corr().sort_values(by='SalePrice', ascending=False)[:10], cbar=True, cmap='viridis', ax=ax)
plt.title('Correlation heatmap of numeric variables to top variables')
plt.show()
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
## Overall quality
o_qual = sns.boxplot(x='OverallQual', y='SalePrice', data=train)
plt.title('Overall Quality vs. Sale Price')
plt.show()
## Above ground living area
gr_area = sns.jointplot(x='SalePrice', y='GrLivArea', data=train)
gr_area.fig.suptitle('Above ground living area\n vs.SalePrice')
gr_area.fig.tight_layout()
gr_area.fig.subplots_adjust(top=0.9)
plt.show()
## Cars per Garage
gar_n = sns.boxplot(x='GarageCars',y='SalePrice', data=train)
plt.suptitle('Cars per Garage')
plt.title('vs. Sale Price')
plt.show()
o_qual.figure.savefig('plots/overall_qual.png')
gr_area.fig.savefig('plots/above_ground.png')
gar_n.figure.savefig('plots/garage_cars.png')
#%%
''' This is a popular dataset so some of the feature engineering and
exploratory analysis are either verbatim or inspired by notebooks Ive browsed
before starting my own analysis.  I will try to be explicit as possible when
using strategies others have used, but may not link a source as some of these
methods are very common when training a model using this data set.
For example, the plots above show that we have some outliers.  Removing these
outliers is very common across the board. They are:
    * A sale with quality = 10 but very low sale price
    * Sales with large living areas but very low sale price
    * Sales with 4 car garages sell lower than those with 3 car garages
Since these dont make sense generally, we will remove these outliers here.  We
will also bear this in mind when deploying the model: we may wish to transform
new data in these ranges.
'''
## Overall quality outliers
train = train.drop(train[(train['SalePrice'] < 200000)
                         & (train['OverallQual'] >= 9)].index)
## Above ground living area outliers
train = train.drop(train[(train['SalePrice'] < 200000)
                         & (train['GrLivArea'] > 4000)].index)
## Four car garage outliers
train = train.drop(train[(train['SalePrice'] < 300000)
                         & (train['GarageCars'] >= 3)].index)
#%%
## Check new distribution of top 3 numerical features
## Overall quality
o_qual = sns.boxplot(x='OverallQual', y='SalePrice', data=train)
plt.title('Overall Quality vs. Sale Price')
plt.show()
## Above ground living area
gr_area = sns.jointplot(x='SalePrice', y='GrLivArea', data=train)
gr_area.fig.suptitle('Above ground living area\n vs.SalePrice')
gr_area.fig.tight_layout()
gr_area.fig.subplots_adjust(top=0.9)
plt.show()
## Cars per Garage
gar_n = sns.boxplot(x='GarageCars',y='SalePrice', data=train)
plt.suptitle('Cars per Garage')
plt.title('vs. Sale Price')
plt.show()
o_qual.figure.savefig('plots/overall_qual.png')
gr_area.fig.savefig('plots/above_ground.png')
gar_n.figure.savefig('plots/garage_cars.png')
#%%
'''
Since we found outliers in features dealing with the garage and above
ground space, and since there are related features in the top ten
correlations, lets inspect those as well.
'''
## Above ground rooms
gr_rms = sns.boxplot(y='SalePrice', x='TotRmsAbvGrd', data=train)
plt.suptitle('Above ground rooms\n vs.SalePrice')
plt.show()
## Cars per Garage
gar_a = sns.jointplot(x='GarageArea',y='SalePrice', data=train)
plt.suptitle('Garage area\n vs. Sale Price')
gar_a.fig.tight_layout()
gar_a.fig.subplots_adjust(top=0.9)
plt.show()
gr_rms.figure.savefig('plots/above_ground_rooms.png')
gar_a.fig.savefig('plots/garage_area.png')
#%%
'''
Here we see that there is an outlier in the Total Rooms Above Ground (=14) and
when the garage area exceeds 1200 feet, we have a few that sold less than
300000.  We can remove these and add to our transformations for new data.
We also see that the variance in sale price is high when we reach the higher
number of rooms above ground.  We can engineer a feature to represent the
sweet spot of rooms above ground as a new variable for modeling.
'''
## Large garage outliers
train = train.drop(train[(train['SalePrice'] < 300000)
                         & (train['GarageArea'] >= 1200)].index)
## Many rooms above ground outliers
train = train.drop(train[(train['SalePrice'] < 300000)
                         & (train['TotRmsAbvGrd'] > 12)].index)
## Binary variable if TotRmsAbvGrd in between 9 & 11
train['sweet_abvgrdrms'] = (train['TotRmsAbvGrd']>=9) & (train['TotRmsAbvGrd'] <= 11)
train['sweet_abvgrdrms'] = train['sweet_abvgrdrms'].astype(int)
#%%
'''
Moving on, as suspected, the variables most correlated with sale price are
those that deal with quality and physical area.
'''
## Get the list of numeric and categorical columns
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
## Lets look at some of the variables by sale price
## We should start with the variables we know relate to outliers
## Basement
bsmt = sns.violinplot(y='SalePrice', x='BsmtFinType1', data=train)
plt.title('Basement Finish vs\n Sale Price')
plt.show()
bsmt.figure.savefig('plots/bsmt_finish.png')

grg = sns.boxplot(y='SalePrice', x='GarageQual', data=train)
plt.title('Garage Quality vs\n Sale Price')
plt.show()
grg.figure.savefig('plots/grg_qual.png')
#%%
'''
Wait, we know that there should be 6 classes for GarageQual, but we only see 5
in our plot.  It is most likely a nan class.
'''
print('Unique GarageQual values:')
print(train['GarageQual'].unique())
'''
Yup.  We'll deal with that later in processing.  For now, we can see that the
classes are well balanced and not skewed, so imputing here should be straight
forward.  Hope this is the case throughout.  Lets continue the exploratory
with visualizations.
'''
#%%
## Sale price distribution by Alley class
alley = sns.displot(x='SalePrice', data=train, hue='Alley', bins=30)
plt.title('Sale Price by Alley class')
plt.show()
alley.fig.savefig('plots/alley_class.png')
## Factorplot of house style
style = sns.catplot(x='HouseStyle', y='SalePrice', data=train)
plt.title('Sale price by house style')
plt.show()
style.fig.savefig('plots/house_style.png')
## Our construct, sweet_abvgrdrms
con = sns.displot(x='SalePrice', hue='sweet_abvgrdrms', data=train)
plt.title('AbvGrdRms construct distribution')
plt.show()
#%%
## Distribution of dwelling types
dwellings = sns.displot(x='MSSubClass', data=train, bins=25)
plt.title('Distribution of Dwelling Types')
plt.show()
## Distribution of dwelling type by sale price
d_sales = sns.boxplot(x='MSSubClass', y='SalePrice', data=train)
plt.title('Dwelling Type vs\n Sale Price')
plt.show()
d_sales.figure.savefig('plots/d_sales.png')
'''
Looks like the vast majority of our sales are the 20 class, or "1-Story 1946
& Newer All Styles".  It also appears like this class has the most variance.
Our most expensive class is 60, or "2 Story 1946 & Newer".  If the dwelling
is older than 1946 or has unfinished space, it looks like they will most
likely be the cheapest.
'''
#%%
## Lets examine numerical columns to inform our normalization strategies
## Distribution of total basement square footage
bsmt_sqf = sns.displot(x='TotalBsmtSF', data=train)
plt.title('Basement Square Footage')
plt.show()
## Jointplot with sales price
bsmt_sq_s = sns.jointplot(x='TotalBsmtSF', y='SalePrice', data=train)
plt.title('Basement Square Footage vs \n SalePrice')
plt.show()
'''
Again we have an outlier here from our exploratory analysis.  There is a 
basement over 6000sq ft in size and sold less than 200000.  Lets remove this
as it will impact our scaler later.
'''
#%%
qual = sns.displot(x='OverallQual', data=train)
plt.title("Overall Quality")
plt.show()
#%%
## Lets move on to missing values
missing = train[train.columns[train.isnull().any().values]]
## Heatmap of columns with missing values
sns.heatmap(missing.isnull().transpose(), cbar=True, cmap = 'viridis')
plt.title('Missing value columns')
plt.show()
'''
Looks like were missing quite a bit of information about the alley and pools.
From the data dictionary, we see that sometimes, missing values represent the
class None. We can see this in the missing values for the basement related
features.  They all line up since the nan here actually represents that the
sale has no basement.
'''
#%%
## Total missing from each column
print(missing.isnull().sum().sort_values(ascending=False))
#%%
