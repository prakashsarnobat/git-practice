# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 21:31:25 2019

@author: Sridhar Sanobat
"""

#filmFile = open("C:/Users/Sridhar Sanobat/Documents/Data Science Examples/ml-100k/u.data", "r")
#movie_data = filmFile.read()
#
#filmFile.close()
#
#print(movie_data)
#print(type(filmFile))
#print(type(movie_data))


import numpy as np
import pandas as pd

movie_data = pd.read_csv('C:/Users/Sridhar Sanobat/Documents/Data Science Examples/ml-100k/u.csv', delimiter = ',', names = ['uid', 'iid', 'rating'])
print(movie_data.head())