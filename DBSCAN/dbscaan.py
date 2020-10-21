# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:50:46 2020

@author: Sharjeel Ahmed
"""



# DBSCAN cluster the data set on the basis of density!
import pandas as pd
import numpy as np

dataset = pd.read_csv('C:/Users/Sharjeel Ahmed/Desktop/ML Algos/ML-Algos/DBSCAN/data.csv')
print (dataset)
#Taking Annual Income and Spending score (indep features) & grouping them into clusters based on density.
X=dataset.iloc[:,[3,4]].values 


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=3, min_samples=4)

#Fitting the model
model =dbscan.fit(X)

labels=model.labels_ #groupig of the clustering results: -1 are the noise 

from sklearn import metrics

#Identifying the points to make our core points
samples_cores=np.zeros_like(labels,dtype=bool) #Make all values false so we can distinguish the clusters
#print(samples_cores) #gives only false values
samples_cores[dbscan.core_sample_indices_]=True #dbscan.core_sample_indices_ gives index values which are clusters
#print(samples_cores) #now it gives true values as well.


#Calculating no of Clusters.! 
clusters= len(set(labels))- (1 if -1 in labels else 0) #i.e 9 clusters form 0 to 8, -1 are noise

#based on the avg mean of the points that are noisy with the indicated points of group clusters
print(metrics.silhouette_score(X,labels))





