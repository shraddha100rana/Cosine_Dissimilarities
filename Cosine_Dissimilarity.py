import pandas as pd
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns

fv = np.load('Feature_Matrix.npy')
periods = f.shape[0]
nodes = f.shape[0]
features = f.shape[0]

#window of average behavior
w = 4
#average behavior matrix
ab = np.zeros((periods,nodes,features))
for t in range (w,periods):
    ab[t,:,:] = np.mean(fv[t-w:t,:,:], axis = 0)

#cosine dissimilarity between feature matrix and average behavior matrix
d = np.zeros((periods,nodes))
for i in range(nodes):
    for t in range (w,periods):
        d[t,i] = spatial.distance.cosine(fv[t,i,:].reshape(features, 1), ab[t,i,:].reshape(features, 1)) 
d = np.nan_to_num(d)

#plot dissimilarity value for time period (y-axis) vs node (x-axis)
plt.figure(figsize = (20, 10))
a = sns.heatmap(d, vmin = 0, vmax = 1, xticklabels = 5, yticklabels = 10)
a.set(title = "Cosine Dissimilarity", xlabel = "Node", ylabel = "Time Period")

#k-means clustering of dissimilarity values
km = KMeans()
a = km.fit(d[w:,:].reshape(-1,1))
#number of clusters
clusters = a.n_clusters
centers = a.cluster_centers_
b = a.labels_.reshape(periods-w,nodes)

k_clusters_bins = np.zeros((clust,3))
for i in range(clusters):
    k_clusters_bins[i,0] = min(d[w:,:].reshape(-1,1)[a.labels_ == i])
    k_clusters_bins[i,1] = max(d[w:,:].reshape(-1,1)[a.labels_ == i])
    k_clusters_bins[i,2] = len(d[w:,:].reshape(-1,1)[a.labels_ == i])
k_clusters_bins = k_clusters_bins[k_clusters_bins[:, 0].argsort()]

ranges = [''] * clusters
for i in range(clusters):
    ranges[i] = str(round(k_clusters_bins[i,0], 2))+"-"+str(round(k_clusters_bins[i,1], 2))
x = np.arange(clusters)
freq = k_clusters_bins[:,2]

#plot cluster histogram of dissimilarity values
plt.figure(figsize = (15, 5))
plt.bar(x, freq, align = 'center')
plt.xticks(x, ranges)
plt.xlabel('Cosine Dissimilarity')
plt.ylabel('Frequency')
plt.show()
