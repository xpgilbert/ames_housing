# Ames Housing
## Regression Modeling for Sale Price
### Includes interactive streamlit application

## Table of Contents

- [Introduction](#introduction)
- [Methods](#methods)
- [Exploratory Data Analysis](#exploratorydataanalysis)

## Introduction
This is a real estate pricing estimator for Ames, Iowa based
on the Ames Housing dataset from [here](https://www.google.com). We will detail the exploratory analysis that influenced our feature engineering, model selection, and hyperparameter tuning.  This dataset is also part of a [kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) and our submission placed in the top 60% of submissions with a
score of 0.15.  On our validation set for which I detail below,
we scored an MAE of 0.01.  

## Methods
All tools specific for this project are built specifically to easily prepare data quickly for our model.  Functions for which can be found in the functions.py file, and most data cleaning is in the load_and_clean.py.  Finally, modeling.py runs our feature engineering and trains the model.  The order to run the python files is:
1. exploratory.py
2. load_and_clean.py
3. modeling.py

Our exploratory analysis includes various visualizations, such as box plots and heatmaps.  Some of these plots revealed both outliers in our data and some expected trends, such as a larger basement led to a higher sale price.  Feature engineering techniques included mean value encoding for categorical variables and mode imputing for missing values since many are categorical.  Mean value was used for missing values in the numeric features.  

## Exploratory Data Analysis
