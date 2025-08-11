import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def boxplot(all_metric_list, list_of_train_sizes, list_of_kernels, representation, yLim = None):
	num_kernels = len(list_of_kernels)

	# Create figure
	fig, ax = plt.subplots(figsize = (15,6))

	# X positions for each kernel
	x_positions = np.arange(num_kernels)
	offset = 0.2  # Adjust for spacing between train sizes

	# Colors for different train sizes
	colors = ["lightblue", "lightgreen", "lightcoral"]
	meanlineprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
	medianprops = dict(linestyle='-', linewidth=1.5, color='black')

	# Plot boxplots for each train size at different offsets
	for i, train_size in enumerate(list_of_train_sizes):
		for j in range(num_kernels):
			ax.boxplot(all_metric_list[i][j], 
					positions=[x_positions[j] + (i - 1) * offset],  # Offset for grouping
					widths=0.15, 
					patch_artist=True,
					sym='',
					medianprops= medianprops,
					meanline=True, showmeans=True, meanprops= meanlineprops,
					boxprops=dict(facecolor=colors[i]))

	# Set x-axis labels
	ax.set_xticks(x_positions)
	ax.set_xticklabels(list_of_kernels)
	
	# Set y-axis limits if provided
	if yLim is not None:
		ax.set_ylim(yLim)

	# Create a legend with colored patches
	legend_patches = [Patch(color=colors[i], label=f"Train size {list_of_train_sizes[i]}") for i in range(3)]
	ax.legend(handles=legend_patches, loc="upper right")

	plt.rc('xtick', labelsize = 16)
	plt.rc('ytick', labelsize = 16)
	plt.rc('legend', fontsize = 16)
	plt.rc('axes', titlesize = 16)

	plt.title(representation, fontsize = 20)
	plt.ylabel('MSSL value')


	# Show the plot
	plt.show()


import pandas as pd
pd.set_option('display.width', 0)
pd.set_option('display.max_columns', None)

def mean_metric_table(all_metric_list, list_of_train_sizes, list_of_kernels, representation):
    num_kernels = len(list_of_kernels)
    
    # Prepare a list to store the formatted mean ± std values
    formatted_values = []

    # Calculate the mean and std for each kernel and train size combination
    for i, train_size in enumerate(list_of_train_sizes):
        row = [train_size]  # Start with the train size column
        for j in range(num_kernels):
            values = all_metric_list[i][j]
            mean_value = np.mean(values)
            std_value = np.std(values)
            row.append(f"{mean_value:.3f} ± {std_value:.2f}")
        formatted_values.append(row)
    
    # Create a DataFrame for the formatted values
    columns = ['Train Size'] + list_of_kernels
    df = pd.DataFrame(formatted_values, columns=columns)

    # Display the table
    print(f"\nMean ± Std MSLL metrics for {representation} Representation:")
    print(df.to_string())

    #return df  # Return the table if needed elsewhere