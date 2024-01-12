import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr
from matplotlib.ticker import ScalarFormatter

# Set global plot configurations
mpl.rcParams["figure.figsize"] = [20, 10]
mpl.rcParams.update({"font.size": 16})
mpl.rc("xtick", labelsize=13)
mpl.rc("ytick", labelsize=13)

def plot_correlations_from_csv(csv_file_path):
    # Read the CSV data into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    fig, axs = plt.subplots(2, 4)
    fig.tight_layout(h_pad=4, w_pad=3)

    # Define the headers and corresponding column names in the DataFrame
    headers = [
        ('Synaptic Diversity', 'Synaptic Diversity'),
        ('Synaptic Saliency', 'Synaptic Saliency'),
        ('Activation Distance', 'Activation Distance'),
        ('Jacobian Score', 'Jacobian Score'),
        ('Head Importance', 'Head Importance'),
        ('Head Confidence', 'Head Confidence'),
        ('Head Softmax Confidence', 'Head Softmax Confidence'),
        ('Number of Parameters', 'Number of Parameters')
    ]

    titles = [header for header, _ in headers]  # Get list of titles from headers

    for i in range(1, len(headers) + 1):
        column_name = headers[i - 1][1]
        data_list = df[column_name].replace([np.inf, -np.inf], np.nan).dropna()
        gt_list = df['GLUE Score'].loc[data_list.index]

        # Set color palette based on the index
        cmap = sns.color_palette("viridis", as_cmap=True)
        if i == 1:
            cmap = sns.color_palette("BuGn", as_cmap=True)
        elif i == 3:
            cmap = sns.color_palette("GnBu", as_cmap=True)
        elif i == 5:
            cmap = sns.color_palette("OrRd", as_cmap=True)
        elif i == 8:
            cmap = sns.color_palette("RdPu", as_cmap=True)
        elif i == 13:
            cmap = sns.color_palette("RdPu", as_cmap=True)

        subplot = axs[(i - 1) // 4, (i - 1) % 4]
        subplot.yaxis.set_major_formatter(ScalarFormatter())
        
        tau, _ = kendalltau(gt_list, data_list)
        rho, _ = pearsonr(gt_list, data_list)
        
        # Create the scatter plot directly with matplotlib to avoid conflicts
        points = subplot.scatter(gt_list, data_list, c=data_list, cmap=cmap, s=10)
        subplot.set_xlabel('GLUE Score')
        subplot.set_ylabel(titles[i - 1])
        subplot.set_title(
            "\n" + titles[i - 1] + "\n τ: {:.3f}    ρ: {:.3f}".format(tau, rho)
        )

        # Add a color bar to show the scale of the colors
        norm = plt.Normalize(data_list.min(), data_list.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # fig.colorbar(sm, ax=subplot, fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for j in range(i, 2 * 4):
        fig.delaxes(axs.flatten()[j])

    plt.show()
    plt.savefig('combined_correlation.png')

# Call the function with the path to your CSV file
plot_correlations_from_csv('./BERT_results_activation.csv')