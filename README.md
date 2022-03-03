# Anomaly detection in dynamic graphs
We use cosine dissimilarity to measure difference in feature matrix at a given time period compared to historic average matrix. 
Then we use k-means clustering on the dissimilarity values.
We use this to determine thresholds of dissimilarity value beyond which a time period-node-feature value would be considered an anomaly.
