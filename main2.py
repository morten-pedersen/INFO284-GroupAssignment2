import matplotlib.pyplot as plotter
import numpy
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
	seeds_dataset = numpy.loadtxt("seeds_dataset.txt")
	seed_labels = seeds_dataset.astype(numpy.int)[:, 7]  # get label from 7th value in dataset
	seed_labels = [label - 1 for label in seed_labels]  # label starts at 0 now
	seeds_dataset = numpy.delete(seeds_dataset, 7, 1)  # remove label from dataset

#functions to calculate Eigenvalues and Eigenvectors of the array, converting the array to a matrix first
#in case we magically find a solution for a rectangle/non-square matrix

#def findEigen():
#	seeds_matrix = numpy.asmatrix(seeds_dataset)
#	eigen_value,eigen_vector = numpy.linalg.eig(seeds_matrix)
#	print("Eigenvalue: ", eigen_value)
#	print("Eigenvector: ", eigen_vector)



def plot(predicted_type, title):
	"""
	This is used to visualize the predicted data
	:param predicted_type:
	:param title: The title of the plot
	:return: a subplot
	"""
	accuracy_score = round(metrics.accuracy_score(seed_labels, predicted_type), 5)
	adjusted_rand_score = round(metrics.adjusted_rand_score(seed_labels, predicted_type), 5)
	predictedPlot = plotter.subplot(2, 1, 1)
	predictedPlot.scatter(seeds_dataset[:, 0], seeds_dataset[:, 1], c = predicted_type, cmap = "rainbow")
	plotter.title(title)
	plotter.title("Accuracy: {}".format(accuracy_score), loc = "right")
	plotter.title("Adj_rand_score: {}".format(adjusted_rand_score), loc = "left")
	actualplot = plotter.subplot(2, 1, 2)
	actualplot.scatter(seeds_dataset[:, 0], seeds_dataset[:, 1], c = seed_labels, cmap = 'rainbow')
	plotter.title('Real')

	return predictedPlot


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


def initiate():
	"""
	Initiate the clustering with kmeans followed by gaussianmixture
	"""
	global seeds_dataset
	global seed_labels
	prepareData()
	pca = PCA(n_components = 5)  # there are 6 features so we choose 6 components as argument
	seeds_dataset = pca.fit_transform(seeds_dataset)  # reducing dimensionality
	gaussian()
	kmeans()


if __name__ == '__main__':
	initiate()
