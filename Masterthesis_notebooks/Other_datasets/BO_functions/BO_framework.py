import warnings
warnings.filterwarnings("ignore") # Turn off Graphein warnings
from typing import Tuple
import os

import time

from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement, MCAcquisitionFunction
from botorch.models.gp_regression import SingleTaskGP
import gpytorch
from mordred import Calculator, descriptors
import numpy as np
from rdkit import Chem
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch

from gauche.dataloader import MolPropLoader
from gauche.dataloader.data_utils import transform_data
from gauche.gp import SIGP, NonTensorialInputs
from gpytorch.kernels import RQKernel
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gauche.kernels.graph_kernels import WeisfeilerLehmanKernel

from BO_functions.prior_selection_alg import *


#################################GP Classes###########################################
class ExactGPModel(SingleTaskGP):
    def __init__(self, kernel, train_x, train_y,):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood = gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        # The binary similarity kernels except Linearkernel don't have an outputscale parameter, so one is added
        if kernel.__name__ != 'LinearKernel':
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel())
        else:
            self.covar_module = kernel() #GPyTorch Linear kernel already has scale parameter
        self.to(train_x)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
###########################################################################################

#################################Aditional functions#######################################
def load_vec_representations(dataset = 'Photoswitch', type='ecfp_fragprints'):
    loader = MolPropLoader(print_info=False)
    loader.load_benchmark(dataset)
    loader.featurize(type)
    X_mol = loader.features
    y_mol = loader.labels
    return X_mol, y_mol

def load_mord_descriptors(dataset = 'Photoswitch'):
    loader = MolPropLoader()
    loader.load_benchmark(dataset)
    # Mordred descriptor computation is expensive
    calc = Calculator(descriptors, ignore_3D=False)
    mols = [Chem.MolFromSmiles(smi) for smi in loader.features]
    X_mordred = [calc(mol) for mol in mols]
    X_mordred = np.array(X_mordred).astype(np.float64) 
    #size length 1425 and elements are real and all the dim values are different
    ###Collect nan indices
    nan_dims = []
    for i in range(len(X_mordred)):
        nan_indices = list(np.where(np.isnan(X_mordred[i, :]))[0])
        for dim in nan_indices:
            if dim not in nan_dims:
                nan_dims.append(dim)
    X_mordred = np.delete(X_mordred, nan_dims, axis=1)
    y_mordred = loader.labels
    return X_mordred, y_mordred

def load_graph_repesentations(dataset = 'Photoswitch'):
    loader = MolPropLoader()
    loader.load_benchmark(dataset)
    loader.featurize("molecular_graphs")
    X_mol = loader.features
    y_mol = loader.labels
    return X_mol, y_mol

###########################################################################################

###############################BO-Framework functions######################################
def optimize_acqf_and_get_observation(acq_func, heldout_inputs, heldout_outputs):
    """
    Optimizes the acquisition function, and returns a new candidate and an observation.
    Args:
        acq_func: Object representing the acquisition function
        heldout_points: Tensor of heldout points
    Returns: new_x, new_obj
    """
    # Loop over the discrete set of points to evaluate the acquisition function
    acq_vals = []
    for i in range(len(heldout_outputs)):
        acq_vals.append(acq_func(heldout_inputs[i].unsqueeze(-2)))

    # observe new values
    acq_vals = torch.tensor(acq_vals)
    best_idx = torch.argmax(acq_vals)
    new_x = heldout_inputs[best_idx].unsqueeze(-2)  
    new_obj = heldout_outputs[best_idx].unsqueeze(-1) 

    # Delete the selected input and value from the heldout set.
    heldout_inputs = torch.cat((heldout_inputs[:best_idx], heldout_inputs[best_idx+1:]), axis=0)
    heldout_outputs = torch.cat((heldout_outputs[:best_idx], heldout_outputs[best_idx+1:]), axis=0)

    return new_x, new_obj, heldout_inputs, heldout_outputs

def update_random_observations(best_random, heldout_inputs, heldout_outputs):
    """
    Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point from the heldout set, observing its value, and updating the list.
    Args:
        best_random: List of best random values observed so far
        heldout_inputs: Tensor of inputs
        heldout_outputs: Tensor of output values
    Returns: best_random, float specifying the objective function value.
    """
    # Take a random sample by permuting the indices and selecting the first element.
    index = torch.randperm(len(heldout_outputs))[0]
    next_random_best = heldout_outputs[index]
    best_random.append(max(best_random[-1], next_random_best))

    # Delete the selected input and value from the heldout set.
    heldout_inputs = torch.cat((heldout_inputs[:index], heldout_inputs[index+1:]), axis=0)
    heldout_outputs = torch.cat((heldout_outputs[:index], heldout_outputs[index+1:]), axis=0)

    return best_random, heldout_inputs, heldout_outputs

def train_model_loop(model, mll, X_train, y_train, train_iterations = 500):
    ###training with gpytorch:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    train_iter = train_iterations
    constraint = model.covar_module.raw_outputscale_constraint

    with torch.no_grad():
        model.covar_module.raw_outputscale.fill_(model.covar_module.raw_outputscale_constraint.inverse_transform(torch.tensor(10.0)))

    print('Start outputscale:', constraint.transform(model.covar_module.raw_outputscale))

    for i in range(train_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()

        optimizer.step()

    print('Iterations %d - Loss: %.3f   raw_outputscale: %.3f  outputscale: %.3f' % (
         train_iter, loss.item(),model.covar_module.raw_outputscale, constraint.transform(model.covar_module.raw_outputscale)
    ))


def PCA_and_transform_data(X_train, y_train, X_test, y_test,pca_dim=50):
    #Standardize the data for better computation
    x_scaler = StandardScaler().fit(X_test)
    X_train, X_test = x_scaler.transform(X_train), x_scaler.transform(X_test)
    y_scaler = StandardScaler().fit(y_test)
    y_train, y_test = y_scaler.transform(y_train), y_scaler.transform(y_test)
    #Perform PCA on the features (trained with X_test that is the biggest)
    pca = PCA(n_components=pca_dim).fit(X_test)
    X_train, X_test = pca.transform(X_train), pca.transform(X_test)
    #Retrutn also the y_scaler to transform back the label result!
    return X_train, X_test, y_train, y_test, y_scaler

###########################################################################################

######################################BO-Framework#########################################
def BO_framework(X_set, y_set, kernel, acquisition_function,
                smart_selection = '', UCB_lambda = 0,
                represtentation ='vector', start_set_size = 0.1, 
                bo_n_iteration = 20, gp_loop_fit = False,
                reproducibility = False, trial =0) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Computes the Bayesian Optimization search process as well as the random search 
    process for bo_n_iterations

    Argumets:
        X_set: Molecule representations 
        y_set: Molecule labels
        kernel: Kernel class used for the GP
        acquisition_function = {EI, UCB, PI}
        smart_selection: algorithm to select the starting training data before BO-Framework (by default rnd)
        representation: type of representation of the inputs! Options={Bin_vec, Mordred, Graphs, Strings}
        start_set_size: porcentage of initial training set
        bo_n_iteration: number of times the BO and RND selection process is done
        gp_loop_fit: Boolean, if true the Framework doesn't use the botorch model fit
        reproducibility: option to reproduce the results, trial is the rnd_seed
        
    Output:
        Tuple of two torch.tensors containing the best obervations of the two serach 
        algorithms (BO-active search & random search) for each iteration.
    '''

    best_BO, best_RND = [],[]
    #Create a random split of staring observations and candidate pool (train_set and test_set)
    rnd_state = trial if reproducibility else None
    n_train_samples = int(len(y_set)*start_set_size)
    rnd_state = trial if reproducibility else None
    match smart_selection:
        case 'least_sim_seq':
            X_tensor = torch.tensor(X_set, dtype=float)
            k = kernel().covar_dist(X_tensor, X_tensor)
            train_idx = least_sim_seq(gram_matrix=k, number_of_samples= n_train_samples,random_state=rnd_state) 
            train_x, test_x, train_y, test_y = split_data(X_set, y_set,train_idx)
        case 'kmeans':
            train_idx = select_kmeans_plusplus(X_set, n_train_samples, random_state=rnd_state)
            train_x, test_x, train_y, test_y = split_data(X_set, y_set,train_idx)
        case 'spectral_clustering':
            X_tensor = torch.tensor(X_set, dtype=float)
            k = kernel().covar_dist(X_tensor, X_tensor)
            train_idx = spectralClustering_rand(gram_matrix=k, n_train_samples=n_train_samples, random_state=rnd_state)
            train_x, test_x, train_y, test_y = split_data(X_set, y_set,train_idx)
        case _:
            train_x, test_x, train_y, test_y = train_test_split(X_set, y_set, train_size=start_set_size, random_state=rnd_state)



    #If we use the mordred descriptors, we pass it trough a transformation and PCA:
    if represtentation == 'Mordred':
        train_x, test_x, _, _, y_scaler = PCA_and_transform_data(train_x, train_y, test_x, test_y,pca_dim=50)
    #Transform the data in torch tensors for its use in GPyTorch
    train_x, test_x, train_y, test_y = map(lambda arr: torch.tensor(arr, dtype=float), [train_x, test_x, train_y, test_y])
    #Since RND and BO will be compared each of them will have a candidate pool (test_x and test_y)
    test_x_rnd, test_y_rnd = test_x, test_y
    #Get the first best observation out of the staring set
    best_observartion_BO, best_observation_RND = torch.max(train_y), torch.max(train_y)
    best_BO.append(best_observartion_BO)
    best_RND.append(best_observation_RND)

    #Initialize the GP model
    gp_model = ExactGPModel(kernel=kernel,
                            train_x=train_x, train_y=train_y)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    #Start BO iteration
    for iteration in range(bo_n_iteration):
        #Fit the model. Two options loop with adam optimizer or default botorch model fit:
        if gp_loop_fit:
            #Training loop:
            train_model_loop(gp_model, mll, train_x, train_y.flatten(), train_iterations = 500)
        else:
            fit_gpytorch_model(mll)
        #Initialize the Acquisition function, different ones:
        match acquisition_function:
            case 'EI':
                acq_fun = ExpectedImprovement(model= gp_model, best_f=(train_y.to(train_y)).max())
            case 'UCB':
                beta = UCB_lambda if (UCB_lambda > 0) else 0.1
                acq_fun = UpperConfidenceBound(model= gp_model, beta=0.1)
            case 'PI':
                acq_fun = ProbabilityOfImprovement(model= gp_model, best_f=(train_y.to(train_y)).max())
            case 'UCB-EI':
                beta = UCB_lambda if (UCB_lambda > 0) else 0.1
                if iteration < 5:
                    acq_fun = UpperConfidenceBound(model= gp_model, beta=beta)
                else:
                    acq_fun = ExpectedImprovement(model= gp_model, best_f=(train_y.to(train_y)).max())
        #BO selection of the best evalutation point and update the train and test set
        new_x_BO, new_y_BO, test_x, test_y = optimize_acqf_and_get_observation(acq_fun,test_x,test_y)
        train_x, train_y = torch.cat([train_x, new_x_BO]), torch.cat([train_y, new_y_BO])
        #Update the best obtained value for BO and up
        best_observartion_BO = torch.max(train_y)
        #RND selection and uptdate the corresponding test_set for RND
        best_RND, test_x_rnd, test_y_rnd = update_random_observations(best_RND, test_x_rnd, test_y_rnd)
        #Add the best observed values from BO and RND to the selection list
        best_BO.append(best_observartion_BO)
        #Initialize the GP again with the new obtained observation from BO to fit it at the start of the loop
        gp_model = ExactGPModel(kernel=kernel,
                        train_x=train_x, train_y=train_y)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    #End of BO loop
    best_BO, best_RND = map(lambda arr: torch.hstack(arr), [best_BO, best_RND])
    return best_BO, best_RND
        
        


