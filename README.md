# Ames Housing
## Regression Modeling for Sale Price

## Table of Contents

- [Introduction](#introduction)
- [Methods](#methods)
- [Exploratory Data Analysis](#exploratorydataanalysis)

## Introduction
This is a real estate pricing estimator for Ames, Iowa based
on the Ames Housing dataset from [here](https://www.google.com). The dataset is distinctly rich and offers a fun, unique data science exercise.  We have access to a wide variety of numerical features, such as the square footage of the finishes basement vs. the total basement square footage, and the data includes categorical features with many relevant classes, such as the type of real-estate feature with over ten classes.  The exercise here is to find train a model that generalizes well despite the richness of our training set.  I am also interested in using this exercise to create a blueprint for future housing price estimators.  As someone who lives in Denver, a housing market that has been off the rails in the last ten years, a local estimator would be a great future project that I can build based on the experience below.

We will detail the exploratory analysis that influenced our feature engineering, model selection, and hyperparameter tuning.  This dataset is also part of a [kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) and our submission placed in the top 60% of submissions with a
score of 0.15.  On our validation set for which I detail below,
we scored an MAE of 0.01.  

## Methods
All tools specific for this project are built specifically to easily prepare data quickly for our model.  Functions for which can be found in the functions.py file, and most data cleaning is in the load_and_clean.py.  Finally, modeling.py runs our feature engineering and trains the model.  The order to run the python files is:
1. exploratory.py
2. load_and_clean.py
3. modeling.py

Our exploratory analysis includes various visualizations, such as box plots and heatmaps.  Some of these plots revealed both outliers in our data and some expected trends, such as a larger basement led to a higher sale price.  Feature engineering techniques included mean value encoding for categorical variables and mode imputing for missing values since many are categorical.  Mean value was used for missing values in the numeric features.  

## Exploratory Data Analysis

To begin, lets explore the shape and size of the data set.  

After some basic exploration of the dataset, our analysis begins with visualizations.  

We first want to see the distribution of SalePrice across the different types of sale classes.  We see below that the 
