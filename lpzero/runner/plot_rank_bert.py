import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy.stats import kendalltau
import matplotlib.colors as mcolors

from lpzero.utils.rank_consistency import spearman

# Set global plot configurations
mpl.rcParams['figure.figsize'] = [13, 6]
mpl.rcParams.update({'font.size': 13})
mpl.rc('xtick', labelsize=13)
mpl.rc('ytick', labelsize=13)

IS_RANK=True

def plot_correlations_from_csv(csv_file_path):
    # Read the CSV data into a DataFrame
    df = pd.read_csv(csv_file_path)

    fig, axs = plt.subplots(2, 4)
    fig.tight_layout(h_pad=2.5, w_pad=3)

    # Define the headers and corresponding column names in the DataFrame
    headers = [
        ('Synaptic Diversity', 'Synaptic Diversity'),
        ('Synaptic Saliency', 'Synaptic Saliency'),
        ('Activation Distance', 'Activation Distance'),
        # ('Jacobian Score', 'Jacobian Score'),
        ('Head Importance', 'Head Importance'),
        ('Head Confidence', 'Head Confidence'),
        ('Head Softmax Confidence', 'Head Softmax Confidence'),
        ('Number of Parameters', 'Number of Parameters'),
        ('LPZero Score', 'lpzero'),
    ]

    # Get list of titles from headers
    titles = [header for header, _ in headers]

    for i in range(len(headers)):
        column_name = headers[i][1]
        data_list = df[column_name].replace([np.inf, -np.inf], np.nan).dropna()
        gt_list = df['GLUE Score'].loc[data_list.index]
        
        if IS_RANK:
            data_list = data_list.rank()
            gt_list = gt_list.rank()

        norm = plt.Normalize(data_list.min(), data_list.max())
        if not IS_RANK:
            cmap = sns.color_palette('viridis')
        else:
            cmap = sns.color_palette('rocket_r')
        
        cmap = mcolors.ListedColormap(cmap)
            
        subplot = axs.flatten()[i]
        subplot.yaxis.set_major_formatter(ScalarFormatter())

        tau, _ = kendalltau(gt_list, data_list)
        rho = spearman(gt_list, data_list)

        data_list = (data_list - min(data_list)) / \
            (max(data_list) - min(data_list))

        # Create the scatter plot directly with matplotlib to avoid conflicts
        subplot.grid(True, linestyle='--', which='major',
                     color='grey', alpha=0.25)
        
        data_list_normalized = (data_list - data_list.min()) / (data_list.max() - data_list.min())
        colors = cmap(data_list_normalized)
        
        subplot.scatter(
            gt_list,
            data_list,
            c=colors,
            s=18,
            edgecolor='black',
            linewidth=0.5,
        )
        subplot.set_xlabel('GLUE Score')
        subplot.set_ylabel(titles[i])
        subplot.set_title(
            # "\n" + titles[i - 1] + "\n
            'τ: {:.3f}    ρ: {:.3f}'.format(tau, rho)
        )

        # Add a color bar to show the scale of the colors
        norm = plt.Normalize(data_list.min(), data_list.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=subplot, fraction=0.046, pad=0.05)

    # Hide any unused subplots
    for j in range(i, 4):
        fig.delaxes(axs.flatten()[j])

    plt.savefig('combined_rank_correlation_3.png', dpi=300, bbox_inches='tight')


# Call the function with the path to your CSV file
plot_correlations_from_csv('./BERT_results_activation_3.csv')
