import numpy as np


def prepareData():
	"""
	This function will prepare the data for
	:return: the prepared data
	"""
	seeds_dataset = np.loadtxt("seeds_dataset.txt")
	seed_labels = seeds_dataset.astype(np.int)[:, 7]  # get label from 7th value in dataset
	seed_labels = [label - 1 for label in seed_labels]  # label starts at 0 now #TODO this might be pointless
	seeds_dataset = np.delete(seeds_dataset, 7, 1)  # remove label from dataset


prepareData()
