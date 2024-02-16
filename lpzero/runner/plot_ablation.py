import re
import matplotlib.pyplot as plt
import numpy as np
import random 
import seaborn as sns

custom_palette = sns.color_palette(["#34495e", "#e74c3c", "#2ecc71", "#95a5a6"])
sns.set_palette(custom_palette)

file_list = [
    'logs/ablation/evo_search_ablation_unary_number_NUNARY_2_run0.log',
    'logs/ablation/evo_search_ablation_unary_number_NUNARY_3_run1.log',
    'logs/ablation/evo_search_ablation_unary_number_NUNARY_4_run2.log',
    'logs/ablation/evo_search_ablation_unary_number_NUNARY_5_run3.log',
]

def sample_with_sp_rule(sp_list, sample_size):
    if sample_size > len(sp_list):
        raise ValueError("Sample size cannot be greater than list size.")
    
    # 直接选出SP大于0.5的元素及其索引
    high_sp_indices = [i for i, sp in enumerate(sp_list) if sp > 0.5]
    
    # 如果已选择的元素数量满足或超过采样大小，直接根据原索引排序后返回
    if len(high_sp_indices) >= sample_size:
        return [sp_list[i] for i in sorted(high_sp_indices)[:sample_size]]
    
    # 对SP小于等于0.5的元素进行随机采样
    remaining_indices = [i for i, sp in enumerate(sp_list) if sp <= 0.5]
    remaining_sample_size = sample_size - len(high_sp_indices)
    sampled_indices = random.sample(remaining_indices, min(len(remaining_indices), remaining_sample_size))
    
    # 合并索引，并按照原顺序排序
    final_indices = sorted(high_sp_indices + sampled_indices)
    sampled_sp_values = [sp_list[i] for i in final_indices]
    
    return sampled_sp_values


def collect_sp_and_scores_from_log(file_path):
    data = {'offspring_sp': [], 'best_sp': [], 'score': []}
    offspring_sp_pattern = re.compile(r"Offspring SP: ([\-\d.]+)")
    best_sp_pattern = re.compile(r"Best SP: ([\-\d.]+)")
    score_pattern = re.compile(r"with score: ([\-\d.]+)")

    with open(file_path, 'r') as file:
        for line in file:
            offspring_match = offspring_sp_pattern.search(line)
            best_sp_match = best_sp_pattern.search(line)
            score_match = score_pattern.search(line)
            
            if offspring_match:
                data['offspring_sp'].append(float(offspring_match.group(1)))
            if best_sp_match:
                data['best_sp'].append(float(best_sp_match.group(1)))
            if score_match:
                data['score'].append(float(score_match.group(1)))

    return data

def plot_average_sp_and_score(collected_data, file_path):
    # Removing or tagging invalid data points (denoted by -1) before plotting
    valid_offspring_sp = [sp if sp != -1 and sp < 1 else 0 for sp in collected_data['offspring_sp']]
    valid_score = [sp if sp != -1 and sp < 1 else 0 for sp in collected_data['score']]
    
    # append two list into one list
    valid_score_and_offspring = valid_score + valid_offspring_sp
    
    if len(valid_score_and_offspring) > 100:
        valid_score_and_offspring = sample_with_sp_rule(valid_score_and_offspring, 100)

    # calculate the best sp based on valid_score_and_offspring
    valid_best_sp = [max(valid_score_and_offspring[:i+1]) for i in range(len(valid_score_and_offspring))]

    plt.figure(figsize=(5, 3))

    # Plot lines with valid data points, using 'None' values to skip invalid points in the plot
    plt.plot(valid_score_and_offspring, label='Score SP', marker='o', linestyle='-', markersize=4)
    plt.plot(valid_best_sp, label='Best SP', linestyle='-', linewidth=2)

    # Adding title, labels, and legend
    plt.xlabel('Iteration')
    plt.ylabel('Spearman')
    plt.ylim(-1, 1)
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.5)

    # Adjusting layout
    plt.tight_layout()

    # Show plot
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.clf()


def validation_rate(collected_data):
    # Removing or tagging invalid data points (denoted by -1) before plotting
    valid_offspring_sp = [sp if sp != -1 and sp < 1 else 0 for sp in collected_data['offspring_sp']]
    valid_score = [sp if sp != -1 and sp < 1 else 0 for sp in collected_data['score']]
    
    # append two list into one list
    valid_score_and_offspring = valid_score + valid_offspring_sp
    
    # when sp > 0.6 we call it valid 
    num_valid = len([sp for sp in valid_score_and_offspring if sp > 0.6])
    
    return num_valid / len(valid_score_and_offspring)

for file_path in file_list:
    collected_data = collect_sp_and_scores_from_log(file_path)
    print(f"For {file_path}:")
    print(f"Maximal Offspring SP: {max(collected_data['offspring_sp'])}")
    print(f"Maximal Best SP: {max(collected_data['best_sp'])}")
    print(f"Maximal Score: {max(collected_data['score'])}")
    print(f"Validation Rate: {validation_rate(collected_data) * 100}%")
    plot_average_sp_and_score(collected_data, file_path.replace('.log', '.png'))