import warnings
warnings.filterwarnings("ignore") # Turn off Graphein warnings

import time

from botorch import fit_gpytorch_model
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

from gpytorch.kernels import RQKernel
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

from matplotlib import pyplot as plt
############################################################################

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, kernel, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # We use the Tanimoto kernel to work with molecular fingerprint representations
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactMordredGPModel(gpytorch.models.ExactGP):
    def __init__(self, kernel, train_x, train_y, likelihood):
        super(ExactMordredGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # We use the RQ kernel to work with Mordred descriptors
        if kernel.__name__ == 'LinearKernel':
            self.covar_module = kernel()
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
############################################################################

def load_vec_representations(type='ecfp_fragprints'):
    loader = MolPropLoader()
    loader.load_benchmark("Photoswitch")
    loader.featurize(type)
    X_mol = loader.features
    y_mol = loader.labels
    return X_mol, y_mol

def load_string_representations(type='bag_of_smiles'):
    loader = MolPropLoader()
    loader.load_benchmark("Photoswitch")
    X_mol = loader.features
    y_mol = loader.labels
    return X_mol, y_mol

def load_mord_descriptors():
    loader = MolPropLoader()
    loader.load_benchmark("Photoswitch")

    # Mordred descriptor computation is expensive
    calc = Calculator(descriptors, ignore_3D=False)
    mols = [Chem.MolFromSmiles(smi) for smi in loader.features]
    t0 = time.time()
    X_mordred = [calc(mol) for mol in mols]
    t1 = time.time()
    print(f'Mordred descriptor computation takes {t1 - t0} seconds')
    X_mordred = np.array(X_mordred).astype(np.float64) 
    #size length 1425 and elements are real and all the dim values are different!!!

    """Collect nan indices"""
    nan_dims = []
    for i in range(len(X_mordred)):
        nan_indices = list(np.where(np.isnan(X_mordred[i, :]))[0])
        for dim in nan_indices:
            if dim not in nan_dims:
                nan_dims.append(dim)

    X_mordred = np.delete(X_mordred, nan_dims, axis=1)
    y_mordred = loader.labels

    return X_mordred, y_mordred

############################################################################
##########################################################################################################################

    
def evaluate_model(X, y, kernel = TanimotoKernel, n_trials = 10, train_set_size = 0.1, metric= 'r2', use_mordred=False, verbose = False):
    """Helper function for model evaluation.

    Args:
        X: n x d NumPy array of inputs representing molecules
        y: n x 1 NumPy array of output labels
        use_mordred: Bool specifying whether the X features are mordred descriptors. If yes, then apply PCA.
    Returns:
        regression metrics and confidence-error curve plot.
    """
    if verbose:
        print(f'\nKernel={kernel.__name__}, n_trials ={n_trials}, test set size={train_set_size}')
    # initialise performance metric lists
    r2_list = []
    rmse_list = []
    mae_list = []
    msll_list = []
    if verbose:
        print('\nBeginning training loop...')

    for i in range(0, n_trials):
        if verbose:
            print(f'Starting trial {i}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_set_size, random_state=i)

        if use_mordred:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            pca_mordred = PCA(n_components=51)
            X_train = pca_mordred.fit_transform(X_train)
            X_test = pca_mordred.transform(X_test)

        #  We standardise the outputs
        _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

        # Convert numpy arrays to PyTorch tensors and flatten the label vectors
        X_train = torch.tensor(X_train.astype(np.float64))
        X_test = torch.tensor(X_test.astype(np.float64))
        y_train = torch.tensor(y_train).flatten()
        y_test = torch.tensor(y_test).flatten()

        # initialise GP likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if use_mordred:
            model = ExactMordredGPModel(X_train, y_train, likelihood)
        else:
            model = ExactGPModel(kernel, X_train, y_train, likelihood)

        # Find optimal model hyperparameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Use the BoTorch utility for fitting GPs in order to use the LBFGS-B optimiser (recommended)
        fit_gpytorch_model(mll)

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # mean and variance GP prediction
        f_pred = model(X_test)

        y_pred = f_pred.mean
        y_var = f_pred.variance

        #compute the msll from the 
        trained_pred_dist = likelihood(model(X_test))
        msll = gpytorch.metrics.mean_standardized_log_loss(trained_pred_dist, y_test).detach()
        
        # Transform back to real data space to compute metrics and detach gradients. Must unsqueeze dimension
        # to make compatible with inverse_transform in scikit-learn version > 1
        y_pred = y_scaler.inverse_transform(y_pred.detach().unsqueeze(dim=1))
        y_test = y_scaler.inverse_transform(y_test.detach().unsqueeze(dim=1))

        # Output Standardised RMSE and RMSE on Train Set
        y_train = y_train.detach()
        y_pred_train = model(X_train).mean.detach()
        train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_rmse = np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_train.unsqueeze(dim=1)),
                                                y_scaler.inverse_transform(y_pred_train.unsqueeze(dim=1))))

        # Compute R^2, RMSE and MAE on Test set
        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        


        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)
        msll_list.append(msll)

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    msll_list = np.array(msll_list)

    if verbose:
        match metric:
            case 'r2':
                print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
                return_val =  r2_list
            case 'RMSE':
                print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
                return_val =  rmse_list
            case 'MAE':
                print("mean MAE: {:.4f} +- {:.4f}".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))
                return_val =  mae_list
            case 'MSLL':
                print("mean MSLL: {:.4f} +- {:.4f}\n".format(np.mean(msll_list), np.std(msll_list)/np.sqrt(len(msll_list))))
                return_val =  msll_list

    match metric:
        case 'r2':
            return_val =  r2_list
        case 'RMSE':
            return_val =  rmse_list
        case 'MAE':
            return_val =  mae_list
        case 'MSLL':
            return_val =  msll_list
    
    return return_val

#############################################################
#############################################################

def mordred_evaluate_model(X, y, kernel = RQKernel, n_trials = 10, train_set_size = 0.1, metric= 'r2', pca_dim = 50, use_mordred=True, verbose = False):
    """Helper function for model evaluation.

    Args:
        X: n x d NumPy array of inputs representing molecules
        y: n x 1 NumPy array of output labels
        use_mordred: Bool specifying whether the X features are mordred descriptors. If yes, then apply PCA.
    Returns:
        regression metrics and confidence-error curve plot.
    """
    if verbose:
        print(f'\nKernel={kernel.__name__}, n_trials ={n_trials}, test set size={train_set_size}')
    # initialise performance metric lists
    r2_list = []
    rmse_list = []
    mae_list = []
    msll_list = []
    if verbose:
        print('\nBeginning training loop...')

    for i in range(0, n_trials):
        if verbose:
            print(f'Starting trial {i}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_set_size, random_state=i)

        if use_mordred:
            scaler = StandardScaler().fit(X)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            pca_mordred = PCA(n_components=pca_dim).fit(X)
            X_train = pca_mordred.transform(X_train)
            X_test = pca_mordred.transform(X_test)

        #  We standardise the outputs
        _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

        # Convert numpy arrays to PyTorch tensors and flatten the label vectors
        X_train = torch.tensor(X_train.astype(np.float64))
        X_test = torch.tensor(X_test.astype(np.float64))
        y_train = torch.tensor(y_train).flatten()
        y_test = torch.tensor(y_test).flatten()

        # initialise GP likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if use_mordred:
            model = ExactMordredGPModel(kernel, X_train, y_train, likelihood)
        else:
            model = ExactGPModel(kernel, X_train, y_train, likelihood)

        # Find optimal model hyperparameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Use the BoTorch utility for fitting GPs in order to use the LBFGS-B optimiser (recommended)
        fit_gpytorch_model(mll)

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # mean and variance GP prediction
        f_pred = model(X_test)

        y_pred = f_pred.mean
        y_var = f_pred.variance

        #compute the msll from the 
        trained_pred_dist = likelihood(model(X_test))
        msll = gpytorch.metrics.mean_standardized_log_loss(trained_pred_dist, y_test).detach()
        
        # Transform back to real data space to compute metrics and detach gradients. Must unsqueeze dimension
        # to make compatible with inverse_transform in scikit-learn version > 1
        y_pred = y_scaler.inverse_transform(y_pred.detach().unsqueeze(dim=1))
        y_test = y_scaler.inverse_transform(y_test.detach().unsqueeze(dim=1))

        # Output Standardised RMSE and RMSE on Train Set
        y_train = y_train.detach()
        y_pred_train = model(X_train).mean.detach()
        train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_rmse = np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_train.unsqueeze(dim=1)),
                                                y_scaler.inverse_transform(y_pred_train.unsqueeze(dim=1))))

        # Compute R^2, RMSE and MAE on Test set
        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        


        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)
        msll_list.append(msll)

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    msll_list = np.array(msll_list)

    if verbose:
        match metric:
            case 'r2':
                print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
                return_val =  r2_list
            case 'RMSE':
                print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
                return_val =  rmse_list
            case 'MAE':
                print("mean MAE: {:.4f} +- {:.4f}".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))
                return_val =  mae_list
            case 'MSLL':
                print("mean MSLL: {:.4f} +- {:.4f}\n".format(np.mean(msll_list), np.std(msll_list)/np.sqrt(len(msll_list))))
                return_val =  msll_list

    match metric:
        case 'r2':
            return_val =  r2_list
        case 'RMSE':
            return_val =  rmse_list
        case 'MAE':
            return_val =  mae_list
        case 'MSLL':
            return_val =  msll_list
    
    return return_val


##########################################################################################################################
##########################################################################################################################

from gauche.kernels.string_kernels.sskkernel import SubsequenceStringKernel



def string_evaluate_model(X, y, kernel = SubsequenceStringKernel, n_trials = 10, train_set_size = 0.1, metric= 'r2', verbose = False):
    """Helper function for model evaluation.

    Args:
        X: n x d NumPy array of inputs representing molecules
        y: n x 1 NumPy array of output labels
        use_mordred: Bool specifying whether the X features are mordred descriptors. If yes, then apply PCA.
    Returns:
        regression metrics and confidence-error curve plot.
    """
    if verbose:
        print(f'\nKernel={kernel.__name__}, n_trials ={n_trials}, test set size={train_set_size}')
    # initialise performance metric lists
    r2_list = []
    rmse_list = []
    mae_list = []
    msll_list = []
    if verbose:
        print('\nBeginning training loop...')

    for i in range(0, n_trials):
        if verbose:
            print(f'Starting trial {i}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_set_size, random_state=i)

        #  We standardise the outputs

        # Convert numpy arrays to PyTorch tensors and flatten the label vectors
        y_train = torch.tensor(y_train).flatten()
        y_test = torch.tensor(y_test).flatten()

        # initialise GP likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(kernel, X_train, y_train, likelihood)

        # Find optimal model hyperparameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Use the BoTorch utility for fitting GPs in order to use the LBFGS-B optimiser (recommended)
        fit_gpytorch_model(mll)

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # mean and variance GP prediction
        f_pred = model(X_test)

        y_pred = f_pred.mean
        y_var = f_pred.variance

        #compute the msll from the 
        trained_pred_dist = likelihood(model(X_test))
        msll = gpytorch.metrics.mean_standardized_log_loss(trained_pred_dist, y_test).detach()
        

        # Output Standardised RMSE and RMSE on Train Set
        y_train = y_train.detach()
        y_pred_train = model(X_train).mean.detach()

        # Compute R^2, RMSE and MAE on Test set
        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        


        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)
        msll_list.append(msll)

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    msll_list = np.array(msll_list)

    if verbose:
        match metric:
            case 'r2':
                print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
                return_val =  r2_list
            case 'RMSE':
                print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
                return_val =  rmse_list
            case 'MAE':
                print("mean MAE: {:.4f} +- {:.4f}".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list))))
                return_val =  mae_list
            case 'MSLL':
                print("mean MSLL: {:.4f} +- {:.4f}\n".format(np.mean(msll_list), np.std(msll_list)/np.sqrt(len(msll_list))))
                return_val =  msll_list

    match metric:
        case 'r2':
            return_val =  r2_list
        case 'RMSE':
            return_val =  rmse_list
        case 'MAE':
            return_val =  mae_list
        case 'MSLL':
            return_val =  msll_list
    
    return return_val