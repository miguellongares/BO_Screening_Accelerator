import numpy as np
import torch


###Algorithms to select the starting points for the training data:

#Simple cover space with molecules more distant to the 'center', problem: molecules get selected many times
def cover_space(gram_matrix, number_of_samples, decrease_factor = 0):
	k = gram_matrix
	center = torch.argmax(torch.sum(k, dim=0)).item()
	samples = [center]
	for n in range(number_of_samples-1):
		similarity = [k[:,samples[i]] for i in range(n+1)]# it could be possible to add a degradation factor where older samples loss importance in the distance 
		similarity = torch.sum(torch.vstack(similarity), dim=0)
		next_sample = torch.argmin(similarity)
		samples.append(next_sample.item())
	return samples


#Simple cover space but without repeat
def prevent_repeat(samples, similarity, random_state = None):
	similarity[samples] = torch.inf
	if random_state is not None:
		np.random.seed(random_state)
		_,index = torch.topk(similarity, k=3, largest=False)
		next_sample = np.random.choice(index)
	else:
		next_sample = torch.argmin(similarity)
	return next_sample

def cover_space_no_repeat(gram_matrix, number_of_samples, random_state = None):
	k = gram_matrix
	center = torch.argmax(torch.sum(k, dim=0)).item()
	samples = [center]
	for n in range(number_of_samples-1):
		similarity = [k[:,samples[i]] for i in range(n+1)]# it could be possible to add a degradation factor where older samples loss importance in the distance 
		similarity = torch.sum(torch.vstack(similarity), dim=0)
		next_sample = prevent_repeat(samples, similarity, random_state)
		samples.append(next_sample.item())
	return samples	

from sklearn.cluster import kmeans_plusplus
#kmeans_plusplus:
def select_kmeans_plusplus(X, number_of_samples, random_state = None):
	_, samples = kmeans_plusplus(X, n_clusters=number_of_samples, random_state= random_state)
	return samples

from sklearn.cluster import spectral_clustering
def spectralClustering_rand(gram_matrix, n_train_samples, random_state = None):
	if random_state is not None:
		np.random.seed(random_state)
	samples = []
	index_mask = spectral_clustering(gram_matrix, n_clusters=n_train_samples)
	for cluster in range(n_train_samples):
		cluster_idx = np.argwhere(index_mask == cluster).flatten()
		#pick one randomly from each cluster:
		sample_idx = np.random.choice(cluster_idx,)
		samples.append(sample_idx)
	return samples


### Extras:

def split_data(X,y, train_idx):
	mask = np.zeros(len(X), dtype=bool)
	mask[train_idx] = True

	X_train = X[mask]
	y_train = y[mask]
	X_test = X[~mask]
	y_test = y[~mask]

	return X_train, X_test, y_train, y_test

