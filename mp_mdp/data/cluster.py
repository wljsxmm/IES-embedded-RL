from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# TODO 聚类后的状态转移矩阵怎么计算 1.先做简单的一维聚类 2.再做二维聚类

# Load the Excel files
df = pd.read_csv('./WindForecast_20220701-20230531.csv')

# use the first column as the row index
df = df.set_index(df.columns[0])

# read the Day-ahead forecast [MW] and the Measured & upscaled [MW] column
df = df[['Day-ahead forecast [MW]', 'Measured & upscaled [MW]']]

# drop any row that contains at least one NaN value
df = df.dropna(axis=0)

# Computing Wind Power Forecast Errors
df['Wind_Prediction_Error'] = df['Day-ahead forecast [MW]'] - df['Measured & upscaled [MW]']


# Prepare the data
# data = df[['Day-ahead forecast [MW]', 'Wind_Prediction_Error']]
data = df['Day-ahead forecast [MW]'].values.reshape(-1, 1)

# It is a good practice to scale the data before clustering
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Compute sum of squared distances for a range of number of cluster
# ssd = []
# range_n_clusters = range(1, 15)  # Adjust this range according to your needs
# for num_clusters in range_n_clusters:
#     kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(data)
#     ssd.append(kmeans.inertia_)
#
# # Plot SSDs for each n_clusters
# plt.plot(range_n_clusters, ssd, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Sum of squared distances')
# plt.title('Elbow Method For Optimal k')
# plt.show()


# Clustering prediction errors
# Specify the number of clusters you want to create
num_clusters = 2

# Select the columns you want to use for clustering
# data = df[['Day-ahead forecast [MW]', 'Wind_Prediction_Error']]

# Initialize KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Fit the model to your data
kmeans.fit(data)

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_
# Sort the cluster centers
sorted_centers = np.sort(cluster_centers, axis=0)
# Calculate the boundaries as the mid-point between cluster centers
boundaries = (sorted_centers[:-1] + sorted_centers[1:]) / 2
# 还原标准化的数据
boundaries = scaler.inverse_transform(boundaries)


# Assign the labels to your data
df['Cluster'] = kmeans.labels_
clusters = {}
for i in set(kmeans.labels_):
    clusters[i] = df[df['Cluster'] == i]

# Plotting a scatterplot of predicted values versus forecast errors
# plt.scatter(df['Wind_Prediction_Error'], df['Day-ahead forecast [MW]'], s=0.5)
# plt.xlabel('Day-ahead forecast [MW]')
# plt.ylabel('Wind prediction error [MW]')
# plt.title('Wind prediction error vs. day-ahead forecast')
# plt.show()