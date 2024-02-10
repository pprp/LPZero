import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns

# Set global plot configurations
mpl.rcParams['figure.figsize'] = [13, 6]
mpl.rcParams.update({'font.size': 13})
mpl.rc('xtick', labelsize=13)
mpl.rc('ytick', labelsize=13)

custom_palette = sns.color_palette(["#9b59b6", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
sns.set_palette(custom_palette)


# 数据准备
search_spaces = ['NB201-CIFAR100', 'FlexiBert']
metrics = ['NWOT', 'Params', 'Synflow', 'SNIP']

# NB201-CIFAR100的ZC指标值
values_nb201 = [0.80, 0.73, 0.76, 0.63]

# FlexiBert的ZC指标值
values_flexibert = [0.331, 0.65, 0.28, 0.43] # to update

# 设置图表大小和子图
fig, ax = plt.subplots(figsize=(6, 4))

# 设置条形图的位置和宽度
bar_width = 0.35
index = np.arange(len(metrics))

# 绘制NB201-CIFAR100的条形图
bars1 = ax.bar(index, values_nb201, bar_width, label='NB201-CIFAR100')

# 绘制FlexiBert的条形图
bars2 = ax.bar(index + bar_width, values_flexibert, bar_width, label='FlexiBert')

# 添加图表标题和坐标轴标签
ax.set_xlabel('ZC')
ax.set_ylabel('Spearman')
# ax.set_title('不同搜索空间的ZC指标比较')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(metrics)
ax.legend(fontsize=10)
# line style in background 
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图表
plt.tight_layout()
plt.savefig('motivation_figure.png', dpi=200)
