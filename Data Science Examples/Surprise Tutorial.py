# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 07:11:46 2019

@author: Sridhar Sanobat
"""

import numpy as np
import pandas as pd


dataset = pd.read_csv('C:/Users/Sridhar Sanobat/Documents/Data Science Examples/filmtrust/ratings.csv', delimiter = ',', names = ['uid', 'iid', 'rating'])
print(dataset.head())
#
lower_rating = dataset['rating'].min()
upper_rating = dataset['rating'].max()
print('Review range: {0} to {1}'.format(lower_rating, upper_rating))
#
import sklearn
import surprise
#
reader = surprise.Reader(rating_scale = (0.5, 4))
data = surprise.Dataset.load_from_df(dataset, reader)
print("Now starting SVD calculation")
#
alg = surprise.SVDpp()
#

train = data.build_full_trainset()
output = alg.fit(train)

print("Displaying training data")
print(train)
print(output) # Extra line added
#

pred = alg.predict(uid = '50', iid = '52')
score = pred.est
print(score)
##
# Get a list of all movie ids
iids = dataset['iid'].unique
print("Displaying a unique list of iids")
print(iids)

# Get a list of iidss that uid 50 has rated
iids50 = dataset.loc[dataset['uid']==50, 'iid']
print("Displaying iids50")
print(iids50)


# Remove the iids that uid50 has rated from the list of all movie ids
#iids_to_pred = np.setdiff1d(iids, iids50)
#print("Displaying iids_to_pred")
#print(iids_to_pred)
#print(type(iids_to_pred))
#iid_column = iids_to_pred.tolist()
#print(type(iid_column))
#print("iids_to_list")
#print(iid_column)
#iid_column


# print ("--------------------------------------------------")
#for item in iids_to_pred:

#for x,y in np.ndenumerate(iids_to_pred):
#    print(type(y))
#    print(y())
#    print("++++++++++++")
#print(iids_to_pred)
##
#testset = [[50, iid, 4.] for iid in iid_column]
#print("Displaying testset")
#print(testset)


dataset50 = dataset.loc[dataset['uid']==50] 
print("Displaying dataset50")
print(dataset50)

datasetNOT50 = dataset.loc[dataset['uid']!=50] 
print("Displaying datasetNOT50")
print(datasetNOT50)
datasetNOT50['uid'] = 50
datasetNOT50['rating'] = 4
print("Displaying datasetNOT50 with uid of 50")
val = datasetNOT50.values
print(val)

iids_to_pred = datasetNOT50['iid']
print("Displaying iids_to_pred")
print(iids_to_pred)

# testset = dataset50.loc[ dataset50['iid'].isin(iids_to_pred) ]
# print("Displaying Testset")
# print(testset)


#testset = [[50, 10, 4],[50, 11, 4]]

# testset = []

#for iid in iids_to_pred[0]:    
#    testset.append([50, iid, 4])
#    print([50, iid, 4])
    

#print(testset) # Extra line added
predictions = alg.test(val)  # datasetNOT50
print("Evaluating on test data")
print(predictions[0])

# Find the index of the maximum predicted rating
pred_ratings = np.array([pred.est for pred in predictions])
print(pred_ratings)

# Find the corresponding iid to recommend
i_max = pred_ratings.argmax()
print("i_max is: ", i_max)

iid = iids_to_pred[i_max]
print("Top item for user 50 has iid {0} with predicted rating {1}".format(iid, pred_ratings[i_max]))

# UNCOMMENT THE BELOW LINES LATER

param_grid = {'lr_all' : [.001, .01], 'reg_all' : [.1, .5]}
gs = surprise.model_selection.GridSearchCV(surprise.SVDpp, param_grid, measures=['rmse','mae'], cv=3)
gs.fit(data)
#Print combination of parameters that give best RMSE score
print(gs.best_params['rmse'])

alg = surprise.SVDpp(lr_all = 0.001) # parameter choices can be added here.
output = surprise.model_selection.cross_validate(alg, data, verbose = True)
