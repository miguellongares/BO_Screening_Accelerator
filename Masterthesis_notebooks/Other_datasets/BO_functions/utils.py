from BO_functions import BO_framework
import torch
import matplotlib.pyplot as plt
import numpy as np

def run_multiple_trials(n_trials, X_set, y_set, kernel,
                        acquisition_function, 
                        smart_selection, representation, 
                        UCB_labda = 0.1, start_set_size = 0.1, 
                        bo_n_iteration = 20, gp_loop_fit = False,
                        reproducibility=False):
    """
    Runs for n times the same BO_sequence where each sequence has a dif starting set!

    Args:

        n_trials: integer
        X_set, y_set: All the data already with the representation
        kernel: Kernel muss fit with the representation
        acquisition_function={'EI', 'UCB', 'PI', 'UCB-EI'}
        UCB_lambda: exploration value. (the higher the more exploration)
        representation={'Bin_vec', 'Mordred', 'Graphs', 'Strings'}
        smart_selection={'least_sim_seq', 'select_kmeans_plusplus', 'spectralClustering_rand'}
        start_set_size: porcent of starting observables
        bo_n_iteration: number of iteratinos on each trial
        gp_loop_fit: Option to use the traditional iteration loop to adjust the parameters in the model

    """
    list_of_BO_sequnces, list_of_RND_sequences = [],[]
    for trial in range(n_trials):
        best_BO, best_RND = BO_framework.BO_framework(X_set, y_set, kernel, 
                                                      acquisition_function, UCB_lambda=UCB_labda,
                                                      smart_selection= smart_selection,
                                                      represtentation = representation, 
                                                      start_set_size = start_set_size , 
                                                      bo_n_iteration = bo_n_iteration, 
                                                      gp_loop_fit = gp_loop_fit,
                                                      reproducibility= reproducibility, trial=trial)#trial functions as the random seed for each run
        list_of_BO_sequnces.append(best_BO)
        list_of_RND_sequences.append(best_RND)
    mean_y_BO = torch.vstack(list_of_BO_sequnces).mean(dim=0)
    std_y_BO = torch.vstack(list_of_BO_sequnces).std(dim=0)
    mean_y_RND = torch.vstack(list_of_RND_sequences).mean(dim=0)
    std_y_RND = torch.vstack(list_of_RND_sequences).std(dim=0)
    return mean_y_BO, std_y_BO, mean_y_RND, std_y_RND
    

def plot_multiple_trial_iteration(data, title_name, x_label, y_label):
    mean_y_BO, std_y_BO, mean_y_RND, std_y_RND = data

    upper_bo_std, lower_bo_std = std_y_BO + mean_y_BO, mean_y_BO - std_y_BO
    upper_rnd_std, lower_rnd_std = mean_y_RND + std_y_RND, mean_y_RND - std_y_RND

    iterations = np.arange(mean_y_BO.shape[0])    

    plt.plot(iterations, mean_y_BO, label = 'BO', color = 'red')
    plt.fill_between(iterations, lower_bo_std, upper_bo_std, alpha = 0.2, color = 'red')
    plt.plot(iterations, mean_y_RND, label = 'RND', color = 'blue')
    plt.fill_between(iterations, lower_rnd_std, upper_rnd_std, alpha = 0.2, color = 'blue')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title_name)
    plt.xticks(list(iterations))
    plt.legend()
    plt.show()    

