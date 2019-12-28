import pandas as pd
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns

#feature vector
fv = np.load('Node_Features_Weekly.npy')
# Size 248 (weeks) x 136 (nodes) x 16 (features)

#past weeks/lags
w = 4

#average behavior
ab = np.zeros((248,136,16))
for t in range (w,248):
    ab[t,:,:] = np.mean(fv[t-w:t,:,:], axis = 0)

#dissimilarity
d = np.zeros((248,136))
#dissimilarity between fv and ab at every time
for i in range(0,136):
    for t in range (w,248):
        d[t,i] = spatial.distance.cosine(fv[t,i,:].reshape(16, 1), ab[t,i,:].reshape(16, 1)) 
d = np.nan_to_num(d)

#plot dissimilarity value for time (y-axis) and node (x-axis)
plt.figure(figsize = (20, 10))
a = sns.heatmap(d, vmin = 0, vmax = 1, xticklabels = 5, yticklabels = 10)
a.set(title = "Cosine Dissimilarity", xlabel = "Node Number", ylabel = "Week Number")

#k-means clustering of all values in d matrix - for finding thresholds
km = KMeans()
a = km.fit(d[w:,:].reshape(-1,1))

#number of clusters
print(a.n_clusters)
print("")

#center of clusters
print(a.cluster_centers_)

b = a.labels_.reshape(248-w,136)

k_clusters_bins = np.zeros((8,3))
for i in range(0,8):
    k_clusters_bins[i,0] = min(d[w:,:].reshape(-1,1)[a.labels_ == i])
    k_clusters_bins[i,1] = max(d[w:,:].reshape(-1,1)[a.labels_ == i])
    k_clusters_bins[i,2] = len(d[w:,:].reshape(-1,1)[a.labels_ == i])
k_clusters_bins = k_clusters_bins[k_clusters_bins[:, 0].argsort()]

ranges = [''] * 8
for i in range(0,8):
    ranges[i] = str(round(k_clusters_bins[i,0], 2))+"-"+str(round(k_clusters_bins[i,1], 2))
x = np.arange(8)
freq = k_clusters_bins[:,2]

plt.figure(figsize = (15, 5))
plt.bar(x, freq, align = 'center')
plt.xticks(x, ranges)
plt.xlabel('Cosine Dissimilarity')
plt.ylabel('Frequency')
plt.title('w = 12 weeks')
plt.show()

#location of anomalies for all combinations of w and tau
locs = [[] for i in range(4)]

#past weeks/lags
w = [4,6,8,12]
#dissimilarity threshold
tau = 0.07

for i in range(len(w)):
    #average behavior
    ab = np.zeros((248,136,16))
    for t in range (w[i],248):
        ab[t,:,:] = np.mean(fv[t-w[i]:t,:,:], axis = 0)

    #dissimilarity
    d = np.zeros((248,136))
    #dissimilarity between fv and ab at every time
    for n in range(0,136):
        for t in range (w[i],248):
            d[t,n] = spatial.distance.cosine(fv[t,n,:].reshape(16, 1), ab[t,n,:].reshape(16, 1)) 
            d = np.nan_to_num(d)
                
    locs[i] = np.argwhere(d >= tau)
    np.savetxt('Location'+str(w[i])+'.csv', locs[i], delimiter=',', fmt='%d')
    print(len(locs[i]))
    
#length of overlap in detected anomaly positions
print(len(np.array([x for x in set(tuple(x) for x in locs[0]) & set(tuple(x) for x in locs[1]) &\
                         set(tuple(x) for x in locs[2]) & set(tuple(x) for x in locs[3])])))
