import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def compute_mean_var_percentage(behavor):
	(mean_gp_list,var_gp_list,mean_selection,var_selection) = behavor

	mean_gp_list = np.array(mean_gp_list)#some values are zero!!
	var_gp_list = np.array(var_gp_list)
	mean_selection = np.array(mean_selection).squeeze()
	var_selection = np.array(var_selection).squeeze()

	mean_gp_list, var_gp_list, mean_selection, var_selection = map(lambda arr: clean_values(arr), [mean_gp_list, var_gp_list, mean_selection, var_selection])

	per_mean = 100 * (mean_selection - mean_gp_list)/mean_gp_list
	per_var = 100 * (var_selection - var_gp_list)/var_gp_list

	return per_mean, per_var

def create_lists():
    all_bo_selection = []
    all_rand_selection = []
    all_acq_vals_mean = []
    all_acq_vals_var = []
    all_perc_mean = []
    all_perc_var = []
    list = (all_bo_selection, all_rand_selection, 
			all_acq_vals_mean, all_acq_vals_var, 
			all_perc_mean, all_perc_var)
    return list

def clean_values(np_array):
	non_zero_mask = np_array != 0
	mean_val = np_array[non_zero_mask].mean()
	np_array[~non_zero_mask] = mean_val
	return np_array

	
def gather_inforamtion(lists, bo_selection, rand_selection, acq_vals_mean, acq_vals_var, behaviour):
	all_bo_selection, all_rand_selection, all_acq_vals_mean, all_acq_vals_var, all_perc_mean, all_perc_var = lists
	all_bo_selection.append(bo_selection)
	all_rand_selection.append(rand_selection)
	all_acq_vals_mean.append(acq_vals_mean)
	all_acq_vals_var.append(acq_vals_var)
	perc_mean, perc_var = compute_mean_var_percentage(behaviour)
	all_perc_mean.append(perc_mean)
	all_perc_var.append(perc_var)
	results = (all_bo_selection, all_rand_selection, 
                all_acq_vals_mean, all_acq_vals_var,
                all_perc_mean, all_perc_var)
	return results



def plot_analysis(results: tuple, n_train_samples: int, kern_name: str='', aqc_fun_name: str = ''):

	(all_bo_selection, all_rand_selection, all_acq_vals_mean, all_acq_vals_var, all_perc_mean, all_perc_var) = results

	mean_y_bo = np.array(all_bo_selection).mean(0)
	std_y_bo = np.array(all_bo_selection).std(0)

	mean_y_rnd = np.array(all_rand_selection).mean(0)
	std_y_rnd = np.array(all_rand_selection).std(0)

	mean_acq_mean_vals = np.array(all_acq_vals_mean).mean(0)
	mean_acq_var_vals = np.array(all_acq_vals_var).mean(0)

	all_perc_mean = np.array(all_perc_mean).mean(0)
	all_perc_mean_std = np.array(all_perc_mean).std(0)

	all_perc_var = np.array(all_perc_var).mean(0)
	all_perc_var_std = np.array(all_perc_var).std(0)

	fig, axes = plt.subplots(1,3, figsize = (18,5))
	fig.suptitle(f"Acq. fun. analysis starting with {n_train_samples} initial observations using {kern_name} and {aqc_fun_name}", fontsize=14)

	###first plot selection comparison 
	axes[0].plot(mean_y_bo, color = 'red', label = 'BO')
	axes[0].fill_between(
		range(len(mean_y_bo)), mean_y_bo - std_y_bo, mean_y_bo + std_y_bo,
		color = 'red', alpha = 0.1 
	)
	axes[0].plot(mean_y_rnd, color = 'blue', label = 'RND')
	axes[0].fill_between(
		range(len(mean_y_bo)), mean_y_rnd - std_y_rnd, mean_y_rnd + std_y_rnd,
		color = 'blue', alpha = 0.1 
	)
	axes[0].set_xticks([i for i in range(len(all_perc_mean)+1)])
	axes[0].set_title('Selection process maximal "y" value')
	axes[0].legend()

	###second plot
	axes[1].plot(mean_acq_mean_vals, color = 'red', label = 'Mean acq.fun. value')

	axes[1].fill_between(
		range(len(mean_acq_mean_vals)), mean_acq_mean_vals - mean_acq_var_vals, mean_acq_mean_vals + mean_acq_var_vals,
		color = 'blue', alpha = 0.1,
		label = 'Variance acqfun value' 
	)
	axes[1].set_xticks([i for i in range(len(all_perc_mean))])
	axes[1].set_title('Acquisition average values')
	axes[1].legend()


	###third plot
	axes[2].plot(all_perc_mean, color = 'red', label = 'Pred mean')
	axes[2].fill_between(
		range(len(all_perc_mean)), all_perc_mean - all_perc_mean_std, all_perc_mean + all_perc_mean_std,
		color = 'blue', alpha = 0.1 
	)
	axes[2].plot(all_perc_var, color = 'black', label = 'Pred uncertainty')
	axes[2].fill_between(
		range(len(all_perc_var)), all_perc_var - all_perc_var_std, all_perc_var + all_perc_var_std,
		color = 'blue', alpha = 0.1 
	)
	axes[2].set_xticks([i for i in range(len(all_perc_mean))])
	axes[2].set_title('Selected sample dif over average')
	axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
	axes[2].legend()

	plt.show()