import matplotlib.pyplot as plotter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def prepareData():
	"""
	This function will prepare the data from
	https://archive.ics.uci.edu/ml/datasets/seeds#
	0. area A,
	1. perimeter P,
	2. compactness C = 4*pi*A/P^2,
	3. length of kernel,
	4. width of kernel,
	5. asymmetry coefficient
	6. length of kernel groove.
	All of these parameters were real-valued continuous.
	:return: a tuple [0]=data [1]=labels
	"""
	seeds_dataset = np.loadtxt("seeds_dataset.txt")

	seed_labels = seeds_dataset.astype(np.int)[:, 7]  # get label from 7th value in dataset
	# seed_labels = [label - 1 for label in seed_labels]  # label starts at 0 now, probably pointless
	seeds_dataset = np.delete(seeds_dataset, 7, 1)  # remove label from dataset
	return seeds_dataset, seed_labels


def kmeans(dataset, data_labels):
	"""
	kmeans is ran with 3 clusters
	:param dataset: the prepared dataset
	:param data_labels: the labels from the dataset
	"""
	kMeans = KMeans(n_clusters = 3, random_state = 2)  # There should be 3 clusters cause there are 3 different seeds
	predicted = kMeans.fit_predict(dataset)  # attempt to predict what class they are
	centroids = kMeans.cluster_centers_  # these are the centroids in the clusters that kMean found
	predictedPlotted = plot(predicted, "kMean:", dataset, data_labels)
	predictedPlotted.scatter(centroids[:, x], centroids[:, y], color = "black", s = 100)
	plotter.show()


def plot(predicted, title, dataset, data_labels):
	"""
	This is used to visualize the predicted data
	:param predicted:
	:param title:
	:param dataset:
	:param data_labels:
	:return:
	"""
	predictedPlot = plotter.subplot(1, 1, 1)
	predictedPlot.scatter(dataset[:, x], dataset[:, y], c = data_labels, cmap = "rainbow")
	plotter.title(title)
	return predictedPlot


x = 0
y = 1
data = prepareData()
seeds_dataset = data[0]
seed_labels = data[1]
pca = PCA(n_components = 7)
seeds_dataset = pca.fit_transform(seeds_dataset)  # reducing dimensionality
kmeans(seeds_dataset, seed_labels)
