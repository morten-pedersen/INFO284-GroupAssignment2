import matplotlib.pyplot as plotter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def prepareData():
	"""
	This function will prepare the data
	1. area A,
	2. perimeter P,
	3. compactness C = 4*pi*A/P^2,
	4. length of kernel,
	5. width of kernel,
	6. asymmetry coefficient
	7. length of kernel groove.
	All of these parameters were real-valued continuous.
	:return: the prepared data
	"""
	return np.loadtxt("seeds_dataset.txt")


def kmeans(dataset):
	kMeans = KMeans(n_clusters = 3, random_state = 2)  # There should be 3 clusters cause there are 3 different seeds
	predicted = kMeans.fit_predict(dataset)  # attempt to predict what class they are
	centroids = kMeans.cluster_centers_  # these are the centroids in the clusters that kMean found
	predictedPlotted = plot(predicted, "kMean:", dataset)
	predictedPlotted.scatter(centroids[:, x], centroids[:, y], marker = "x", s = 100, zorder = 10)
	plotter.show()


def plot(predicted, title, dataset):
	predictedPlot = plotter.subplot(2, 1, 1)
	predictedPlot.scatter(dataset[:, x], dataset[:, y], c = "red", cmap = "rainbow")
	plotter.title(title)
	return predictedPlot


x = 0
y = 1
dataset = prepareData()
pca = PCA(n_components = 7)
dataset = pca.fit_transform(dataset)  # reducing dimensionality
kmeans(dataset)
