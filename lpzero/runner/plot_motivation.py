import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns

# Set global plot configurations
# mpl.rcParams['figure.figsize'] = [13, 6]
mpl.rcParams.update({'font.size': 14})
mpl.rc('xtick', labelsize=9)
mpl.rc('ytick', labelsize=13)

custom_palette = sns.color_palette(["#9b59b6", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
sns.set_palette(custom_palette)


# # 数据准备
# search_spaces = ['NB201-CIFAR100', 'FlexiBert']
# metrics = ['NWOT', 'Params', 'Synflow', 'SNIP']

# # NB201-CIFAR100的ZC指标值
# values_nb201 = [0.80, 0.73, 0.76, 0.63]

# # FlexiBert的ZC指标值
# values_flexibert = [0.331, 0.65, 0.28, 0.43] # to update

# # 设置图表大小和子图
# fig, ax = plt.subplots(figsize=(6, 4))

# # 设置条形图的位置和宽度
# bar_width = 0.35
# index = np.arange(len(metrics))

# # 绘制NB201-CIFAR100的条形图
# bars1 = ax.bar(index, values_nb201, bar_width, label='NB201-CIFAR100')

# # 绘制FlexiBert的条形图
# bars2 = ax.bar(index + bar_width, values_flexibert, bar_width, label='FlexiBert')

# # 添加图表标题和坐标轴标签
# ax.set_xlabel('ZC')
# ax.set_ylabel('Spearman')
# # ax.set_title('不同搜索空间的ZC指标比较')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(metrics)
# ax.legend(fontsize=10)
# # line style in background 
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# # 显示图表
# plt.tight_layout()
# plt.savefig('motivation_figure.png', dpi=200)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# First, we prepare the data with abbreviations
data_spearman_full = {
    'Proxy': [
        'Activation Distance', 'Jacobian Cosine', 'Attention Importance', 'Head Importance',
        'SNIP', 'Synaptic Diversity', 'GraSP', 'GradNorm', 'Fisher', 'Synaptic Saliency',
        'Head Confidence', 'Synflow', 'LogSynflow', 'No. Params.', 'Attention Confidence', 'LPZero (Ours)'
    ],
    'Spearman’s ρ': [
        0.123, 0.149, 0.162, 0.171, 0.173, 0.175, 0.179, 0.197, 0.209, 0.266,
        0.364, 0.471, 0.491, 0.590, 0.666, 0.748
    ]
}

# Creating DataFrame from the data
df_spearman_full = pd.DataFrame(data_spearman_full)

# Define abbreviations for proxies
abbreviations = {
    'Synaptic Diversity': 'Syn. Div.',
    'Synaptic Saliency': 'Syn. Sal.',
    'Activation Distance': 'Act. Dist.',
    'Jacobian Cosine': 'Jac. Cos.',
    'Attention Confidence': 'Att. Conf.',
    'Attention Importance': 'Att. Imp.',
    'Head Importance': 'Head Imp.',
    'Head Confidence': 'Head Conf.',
    'No. Params.': 'No. Params',
    'LPZero (Ours)': 'LPZero'
}

# Apply abbreviations to the DataFrame
df_spearman_full['Proxy Abbreviated'] = df_spearman_full['Proxy'].map(abbreviations).fillna(df_spearman_full['Proxy'])

# Sort the DataFrame based on 'Spearman’s ρ' scores
df_spearman_sorted = df_spearman_full.sort_values('Spearman’s ρ').reset_index(drop=True)

# Find the baseline score for 'No. Params' in Spearman’s ρ
spearman_baseline_full = df_spearman_sorted[df_spearman_sorted['Proxy'] == 'No. Params.']['Spearman’s ρ'].values[0]

# Create the sorted bar plot with abbreviated x labels
plt.figure(figsize=(6, 3.5))
bar_plot = sns.barplot(x='Proxy Abbreviated', y='Spearman’s ρ', data=df_spearman_sorted, color='salmon')
plt.axhline(spearman_baseline_full, color='red', linewidth=1.5, label='Baseline')
# plt.title('Spearman’s ρ Scores (Sorted)')
plt.xticks(rotation=45, ha='right')  # Rotate the x labels by 45 degrees
plt.ylabel('Spearman’s ρ')
plt.legend()
plt.ylim(0, max(df_spearman_sorted['Spearman’s ρ']) + 0.1)  # Add some space above the highest bar
bar_plot.set(xlabel=None)
# Show the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('motivation_figure.png', dpi=300, bbox_inches='tight')
