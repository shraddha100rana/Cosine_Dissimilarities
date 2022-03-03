# Analysis of dynamic graphs

In Cosine_Dissimilarity.py we use cosine dissimilarity to measure the difference between the graph's behavior at a given time period and its historic average behavior. Then, we use k-means clustering to find similar groups of dissimilarity values. These clusters can be used to find threshold of dissimilarity value beyond which the (time period, node, feature value) would be considered an anomaly.

In Spatial_Relationship_Discovery.py we use spatio-temporal neural networks to learn spatial relationship of nodes in dynamic graphs, given input and output temporal features of nodes.

In Time_Series_Forecasting.py we use spatio-temporal neural networks to learn temporal relationships of dynamic graphs.

In Morans_Index.py we calculate the Moran's I of dynamic graphs that measures their spatial autocorrelation.
