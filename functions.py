#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:05:11 2021

@author: Gilly
"""
import numpy as np
import pandas as pd

class Load():
    def __init__(self, data, none_cols = []):
        '''
        Parameters
        ----------
        data : pandas DataFrame
            features on which to train the model later.
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