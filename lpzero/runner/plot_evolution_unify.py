import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Set global plot configurations
# mpl.rcParams.update({'font.size': 13})
# mpl.rc('xtick', labelsize=13)
# mpl.rc('ytick', labelsize=13)
mpl.rcParams.update({'font.size': 14})
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)


custom_palette = sns.color_palette(["#9b59b6", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
sns.set_palette(custom_palette)

def plot_logs_in_lineplot(log_files, src_dir):
    plt.figure(figsize=(10, 5))
    
    for log_file in log_files:
        idxs, kds = [], []
        idx = 0
        full_path = os.path.join(src_dir, log_file)
        with open(full_path, 'r') as file:
            for line in file:
                if 'KD' in line:
                    kd = re.search(r'KD: ([\d.]+)', line)
                    if kd:
                        kd = kd.group(1)
                        idxs.append(idx)
                        kds.append(float(kd))
                        idx += 1
        
        # 绘制每个日志文件的线条，并使用文件名作为标签
        sns.lineplot(x=idxs, y=kds, label=log_file.replace('.log', ''))
    
    plt.xlabel('Iteration')
    plt.ylabel('SP')
    plt.grid()
    plt.legend()
    plt.savefig('./logs/combined_kds_bak.png', dpi=300, bbox_inches='tight')

def plot_logs_in_lineplot_SP(log_files, src_dir):
    plt.figure(figsize=(6, 4))
    
    for log_file in log_files:
        idxs, kds = [], []
        idx = 0
        best_kd = -1
        full_path = os.path.join(src_dir, log_file)
        with open(full_path, 'r') as file:
            for line in file:
                if 'SP' in line:
                    kd = re.search(r'SP: ([\d.]+)', line)
                    if kd:
                        kd = kd.group(1)
                        if float(kd) > best_kd:
                            best_kd = float(kd)
                        if idx % 2 == 0:
                            idxs.append(idx)
                            kds.append(best_kd)
                        idx += 1
                if 'KD' in line:
                    kd = re.search(r'KD: ([\d.]+)', line)
                    if kd:
                        kd = kd.group(1)
                        idxs.append(idx)
                        kds.append(float(kd))
                        idx += 1
        
        # 绘制每个日志文件的线条，并使用文件名作为标签
        sns.lineplot(x=idxs, y=kds, label=log_file.replace('.log', '').replace('w', 'w/'), linewidth=2.5)
    
    plt.xlabel('Iteration')
    plt.ylabel('Spearman')
    plt.xlim(0, 1000)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('./logs/combined_final_popu.png', dpi=300, bbox_inches='tight')

def traverse_dirs(src_dir):
    log_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.log'):
                log_files.append(os.path.join(root, file).replace(src_dir + '/', ''))
    plot_logs_in_lineplot_SP(log_files, src_dir)

src_dir = './logs/evolution/popu'
traverse_dirs(src_dir)
