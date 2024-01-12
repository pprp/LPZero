import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lpzero.metrics.cluster_correlation_index import measure_cluster_corr_index
from lpzero.metrics.mutual_info import measure_mutual_information
from lpzero.metrics.silhouetee import measure_silhouette


def load_data(path):
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f'Error loading data: {e}')
        return None


def calculate_metrics(mem_dict):
    record_mi = []
    record_s = []
    record_sp = []
    record_cci = []

    for key, data in mem_dict.items():
        print(f'Processing {key}...')
        gt_list, zc_list = data['gt_list'], data['zc_list']
        mi_score = measure_mutual_information(gt_list, zc_list)
        s_avg = measure_silhouette(gt_list, zc_list, 3)
        cci_score = measure_cluster_corr_index(gt_list, zc_list, 1, 3)

        print(
            f'MI Score: {mi_score}, Silhouette Avg: {s_avg}, CCI Score: {cci_score}')
        print('===')

        record_mi.append(mi_score)
        record_s.append(s_avg)
        record_sp.append(data['sp'])
        record_cci.append(cci_score)

    return record_mi, record_s, record_sp, record_cci


def improved_plot_distribution(data, title, filename):
    sns.set_theme(style='whitegrid')
    num_bins = min(len(set(data)), 50)
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=num_bins, kde=False)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.clf()


def plot_distribution(data, title, filename):
    sns.set_theme(style='whitegrid')
    sns.displot(data)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.clf()


def plot_correlations(data):
    sns.set_theme(style='ticks')
    pairplot = sns.pairplot(data)
    pairplot.fig.suptitle(
        'Pairwise Plots of MI, Silhouette, Spearman Correlation, and CCI', y=1.02
    )
    plt.savefig('correlations.png')
    plt.clf()


def main():
    path = './mem_dict.json'
    mem_dict = load_data(path)

    if mem_dict:
        record_mi, record_s, record_sp, record_cci = calculate_metrics(
            mem_dict)
        record_s = (record_s - np.mean(record_s)) / np.std(
            record_s
        )  # Normalizing record_s

        plot_distribution(
            record_mi, 'Distribution of Mutual Information Scores', 'mi.png'
        )
        improved_plot_distribution(
            record_s, 'Normalized Distribution of Silhouette Scores', 's.png'
        )
        plot_distribution(
            record_sp, 'Distribution of Spearman Correlation Scores', 'sp.png'
        )
        data = pd.DataFrame(
            {
                'Mutual Information': record_mi,
                'Normalized Silhouette': record_s,
                'Spearman Correlation': record_sp,
                'CCI': record_cci,
            }
        )
        plot_correlations(data)

        # max mutual and min mutual information
        max_mi = max(record_mi)
        min_mi = min(record_mi)
        mean_mi = np.mean(record_mi)
        print(f'Max MI: {max_mi}, Min MI: {min_mi}, Mean MI: {mean_mi}')

        # max and min sp correlation
        max_sp = max(record_sp)
        min_sp = min(record_sp)
        mean_sp = np.mean(record_sp)
        print(f'Max SP: {max_sp}, Min SP: {min_sp}, Mean SP: {mean_sp}')

        # max and min silhouette
        max_s = max(record_s)
        min_s = min(record_s)
        mean_s = np.mean(record_s)
        print(f'Max S: {max_s}, Min S: {min_s}, Mean S: {mean_s}')


if __name__ == '__main__':
    main()
