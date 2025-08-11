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

from matplotlib import pyplot as plt

class GraphGP(SIGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.covariance = WeisfeilerLehmanKernel(node_label="element")

    def forward(self, x):
        mean = self.mean(torch.zeros(len(x), 1)).float()
        covariance = self.covariance(x)

        # for numerical stability
        jitter = max(covariance.diag().mean().detach().item() * 1e-4, 1e-4)
        covariance += torch.eye(len(x)) * jitter
        return gpytorch.distributions.MultivariateNormal(mean, covariance)
    
def UpCoBo(model: GraphGP, X_test, beta):
    mean, sigma = model(X_test).mean, model(X_test).variance
    acq_vals = mean + beta*sigma**(1/2)
    return acq_vals
    

def optimize_acqf_and_get_observation(model, heldout_inputs, heldout_outputs):
    """
    Optimizes the acquisition function, and returns a new candidate and an observation.
    Args:
        acq_func: Object representing the acquisition function
        heldout_points: Tensor of heldout points
    Returns: new_x, new_obj
    """
    # Loop over the discrete set of points to evaluate the acquisition function at.
    acq_vals = UpCoBo(model, heldout_inputs, beta=0.1)
    best_idx = torch.argmax(acq_vals)
    new_x = heldout_inputs.data[best_idx]
    new_obj = heldout_outputs[best_idx].unsqueeze(-1)  # add output dimension

    # Delete the selected input and value from the heldout set.
    heldout_inputs.data.remove(new_x)
    heldout_outputs = torch.cat((heldout_outputs[:best_idx], heldout_outputs[best_idx+1:]), axis=0)

    return new_x, new_obj, heldout_inputs, heldout_outputs

def update_random_observations(best_random, heldout_outputs):
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
    heldout_outputs = torch.cat((heldout_outputs[:index], heldout_outputs[index+1:]), axis=0)

    return best_random, heldout_outputs




def BO_framework_graph(X_set, y_set, acquisition_function, start_set_size = 0.1, bo_n_iteration = 20, gp_loop_fit = False) -> Tuple[torch.Tensor, torch.Tensor]:###Maybe implement the random seed number!
    '''
    Computes the Bayesian Optimization search process as well as the random search 
    process for bo_n_iterations

    Argumets:
        X_set: Molecule representations 
        y_set: Molecule labels
        kernel: Kernel class used for the GP
        acquisition_function = {EI, UCB, PI}
        representation: type of representation of the inputs! Options={Bin_vec, Mordred, Graphs, Strings}
        start_set_size: porcentage of initial training set
        bo_n_iteration: number of times the BO and RND selection process is done
        gp_loop_fit: Boolean, if true the Framework doesn't use the botorch model fit
        
    Output:
        Tuple of two torch.tensors containing the best obervations of the two serach 
        algorithms for each iteration
    '''

    best_BO, best_RND = [],[]
    #Create a random split of staring observations and candidate pool (train_set and test_set)
    train_x, test_x, train_y, test_y = train_test_split(X_set, y_set, train_size=start_set_size)
    #Transform the data in torch tensors for its use in GPyTorch
    train_x = NonTensorialInputs(train_x)
    test_x = NonTensorialInputs(test_x)
    train_y = torch.tensor(train_y).flatten().float()
    test_y = torch.tensor(test_y).flatten().float()
    #Since RND and BO will be compared each of them will have a candidate pool (test_x and test_y)
    test_x_rnd, test_y_rnd = test_x, test_y
    #Get the first best observation out of the staring set
    best_observartion_BO, best_observation_RND = torch.max(train_y), torch.max(train_y)
    best_BO.append(best_observartion_BO)
    best_RND.append(best_observation_RND)

    #Initialize the GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = GraphGP(train_x, train_y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    #Start BO iteration
    for iteration in range(bo_n_iteration):
        #Fit the model. Two options loop with adam optimizer or default botorch model fit:

        fit_gpytorch_model(mll)

        #BO selection of the best evalutation point and update the train and test set
        new_x_BO, new_y_BO, test_x, test_y = optimize_acqf_and_get_observation(gp_model,test_x,test_y)
        train_x.data = train_x.data + [new_x_BO]
        train_y = torch.cat([train_y, new_y_BO])
        #Update the best obtained value for BO and up
        best_observartion_BO = torch.max(train_y)
        #RND selection and uptdate the corresponding test_set for RND
        best_RND, test_y_rnd = update_random_observations(best_RND, test_y_rnd)
        #Add the best observed values from BO and RND to the selection list
        best_BO.append(best_observartion_BO)
        #Initialize the GP again with the new obtained observation from BO to fit it at the start of the loop
        gp_model = GraphGP(train_x, train_y, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    #End of BO loop
    best_BO, best_RND = map(lambda arr: torch.hstack(arr), [best_BO, best_RND])
    return best_BO, best_RND