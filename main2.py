import numpy as np


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
	seeds_dataset = np.loadtxt("seeds_dataset.txt")
	

	# seed_labels = seeds_dataset.astype(np.int)[:, 7]  # get label from 7th value in dataset
	# seed_labels = [label - 1 for label in seed_labels]  # label starts at 0 now #TODO this might be pointless
	# seeds_dataset = np.delete(seeds_dataset, 7, 1)  # remove label from dataset


prepareData()
