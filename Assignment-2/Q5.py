# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:47:50 2020

@author: DELL
"""

import numpy as np
import scipy.io
from sklearn import datasets
from sklearn.model_selection import train_test_split    
import pandas as pd

mat = scipy.io.loadmat("Datasets/data5.mat")
mat_df = pd.DataFrame(mat['x'])
class_array = pd.DataFrame.to_numpy(mat_df)

x_train, x_test, y_train, y_test = train_test_split(class_array[:, :-1], class_array[:, -1], test_size=0.7)

