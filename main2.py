import matplotlib.pyplot as plotter
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


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
	"""
	global seeds_dataset
	global seed_labels
	seeds_dataset = np.loadtxt("seeds_dataset.txt")
	seed_labels = seeds_dataset.astype(np.int)[:, 7]  # get label from 7th value in dataset
	seed_labels = [label - 1 for label in seed_labels]  # label starts at 0 now
	seeds_dataset = np.delete(seeds_dataset, 7, 1)  # remove label from dataset


def kmeans():
	"""
	kmeans is ran with 3 clusters
	"""
	kMeans = KMeans(algorithm = "auto", n_clusters = 3,
					random_state = 8)  # There should be 3 clusters cause there are 3 different seeds
	predicted = kMeans.fit_predict(seeds_dataset)  # attempt to predict what class they are
	centroids = kMeans.cluster_centers_  # these are the centroids in the clusters that kMean found
	predictedPlotted = plot(predicted, "kMean:")
	predictedPlotted.scatter(centroids[:, 0], centroids[:, 1], color = "black", s = 100, zorder = 10)
	plotter.show()


def gaussian():
	"""
	This will run Gaussian Mixture with 3 components cause there are 3 kinds of seeds,
	 random_state 62 seems to work pretty well
	"""
	gaussianMixture = GaussianMixture(n_components = 3, random_state = 62).fit(seeds_dataset)
	# Save predicted classes to list
	pred_type = gaussianMixture.predict(seeds_dataset)
	# Fill a plot with predicted types. Add a title
	plot(pred_type, 'GaussianMixture')
	plotter.show()


def plot(predicted_type, title):
	"""
	This is used to visualize the predicted data
	:param predicted_type:
	:param title:

	:return:
	"""
	accuracy_score = round(metrics.accuracy_score(seed_labels, predicted_type), 5)
	adjusted_rand_score = round(metrics.adjusted_rand_score(seed_labels, predicted_type), 5)
	predictedPlot = plotter.subplot(2, 1, 1)
	predictedPlot.scatter(seeds_dataset[:, 0], seeds_dataset[:, 1], c = predicted_type, cmap = "rainbow")
	plotter.title(title)
	plotter.title("Accuracy: {}".format(accuracy_score), loc = "right")
	plotter.title("ARI: {}".format(adjusted_rand_score), loc = "left")

	actualplot = plotter.subplot(2, 1, 2)
	actualplot.scatter(seeds_dataset[:, 0], seeds_dataset[:, 1], c = seed_labels, cmap = 'rainbow')
	plotter.title('Real')

	return predictedPlot


def initiate():
	global seeds_dataset
	global seed_labels
	prepareData()
	pca = PCA(n_components = 5)
	seeds_dataset = pca.fit_transform(seeds_dataset)  # reducing dimensionality
	kmeans()


# gaussian()


if __name__ == '__main__':
	initiate()
