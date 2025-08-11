import gpytorch
import numpy as np
import torch
from botorch import fit_gpytorch_model
from gpytorch.models import ExactGP
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from gauche.dataloader import MolPropLoader
from gauche.dataloader.data_utils import transform_data
from gauche.gp import SIGP, NonTensorialInputs
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gauche.kernels.graph_kernels import WeisfeilerLehmanKernel

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
    

def eval_graph_GP(X, y, train_set_size = 0.1, n_trials = 10, metric = 'r2'):

    r2_list = []
    rmse_list = []
    mae_list = []
    msll_list = []

    for i in range(0, n_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_set_size, random_state=i)
        _, y_train, _, y_test, y_scaler = transform_data(np.zeros_like(y_train), y_train, np.zeros_like(y_test), y_test)


        X_train = NonTensorialInputs(X_train)
        X_test = NonTensorialInputs(X_test)
        y_train = torch.tensor(y_train).flatten().float()
        y_test = torch.tensor(y_test).flatten().float()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GraphGP(X_train, y_train, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        fit_gpytorch_model(mll)

        model.eval()
        likelihood.eval()

        y_pred = y_scaler.inverse_transform(model(X_test).mean.detach().unsqueeze(dim=1))
        

        trained_pred_dist = likelihood(model(X_test))
        msll = gpytorch.metrics.mean_standardized_log_loss(trained_pred_dist, y_test).detach()

        y_test = y_scaler.inverse_transform(y_test.detach().unsqueeze(dim=1))

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