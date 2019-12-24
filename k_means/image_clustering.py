from sklearn import cluster
from sklearn.cluster import KMeans
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

catImage = imread('dog.jpg')
imgplot = plt.imshow(catImage)

x, y, z = catImage.shape
image_2d = catImage.reshape(x*y, z)
n=4
kmeans_cluster = cluster.KMeans(n_clusters=n)
kmeans_cluster.fit(image_2d)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_
plt.imshow((cluster_centers[cluster_labels].reshape(x, y, z) * 255).astype(np.uint8))
plt.show()