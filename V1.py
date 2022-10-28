# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 20:31:01 2022

@author: hugos
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

#Load data from csv and store it in a dataframe
data = pd.read_csv("pima_indian_diabetes.csv") #We can change this database

# We show the first individuals
data.head() # Return the first n rows