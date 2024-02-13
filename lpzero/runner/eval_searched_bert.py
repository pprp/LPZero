import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from lpzero.predictor.measures.fisher import compute_fisher_per_weight
from lpzero.predictor.measures.snip import compute_snip_per_weight 
from lpzero.predictor.measures.grasp import compute_grasp_per_weight
from lpzero.predictor.measures.synflow import compute_synflow_per_weight, compute_logsynflow_per_weight
from lpzero.model.flexibert.modeling_electra import ElectraLayer, ElectraModel
from lpzero.predictor.measures.grad_norm import get_grad_norm_arr
from lpzero.model.flexibert.modeling_electra import (
    ElectraConfig,
    ElectraModel,
)
from lpzero.runner.evo_search_bert import all_same, is_anomaly, parse_args
from lpzero.structures import GraphStructure, LinearStructure, TreeStructure
from lpzero.utils.rank_consistency import spearman, kendalltau
from lpzero.utils.preprocess_openwebtext import generate_inputs
from lpzero.predictor.measures.activation_distance import compute_act_dist

configs = []
with open('./data/BERT_benchmark.json', 'r') as f:
    configs = json.load(f)

args = parse_args()


def fitness_spearman(inputs, structure, device=None, num_sample=500):
    """structure is belong to popultion."""
    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    if structure.sp_score != -1:
        return structure.sp_score

    gt_score = []
    zc_score = []

    df = pd.read_csv('./BERT_results_activation.csv')

    for i in tqdm(range(num_sample)):
        nas_config = configs[i]['hparams']['model_hparam_overrides']['nas_config']

        gt = configs[i]['scores']['glue']
        # build a new model
        config = ElectraConfig(
            nas_config=nas_config,
            num_hidden_layers=len(nas_config['encoder_layers']),
            output_hidden_states=True,
        )
        model = ElectraModel(config)
        model.to(device)
        inputs.to(device)
        # compute zc with the given structure
        zc = structure(model, inputs)

        if is_anomaly(zc):
            return -1

        # early exit
        if len(zc_score) > 3 and all_same(zc_score):
            return -1

        zc_score.append(zc)
        gt_score.append(gt)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # gc.collect()

    # TODO add inf check
    try:
        if len(zc_score) <= 1 or np.isnan(spearman(gt_score, zc_score)):
            return -1
    except TypeError as e:
        import pdb

        pdb.set_trace()

    df['lpzero'] = zc_score
    df.to_csv('./BERT_results_activation_3.csv', index=False)

    # release memory
    del inputs
    torch.cuda.empty_cache()
    # gc.collect()

    sp = spearman(gt_score, zc_score)
    kd = kendalltau(gt_score, zc_score)
    
    print(f'Spearman of {structure} is {sp}')
    print(f'Kendalltau of {structure} is {kd}')
    
    if structure.sp_score == -1:
        structure.sp_score = sp

    # plot the result
    plt.figure()
    # z-score for zc_score
    # zc_score = (zc_score - np.mean(zc_score)) / np.std(zc_score)
    # min-max scale
    zc_score = (zc_score - np.min(zc_score)) / \
        (np.max(zc_score) - np.min(zc_score))
    # filter pairs that zc_score is larger than 0.8
    # gt_score = np.array(gt_score)[np.where(zc_score > 0.8)]
    # zc_score = np.array(zc_score)[np.where(zc_score > 0.8)]
    plt.scatter(gt_score, zc_score, marker='o', color='red')
    plt.xlabel('Ground Truth')
    plt.ylabel('Zero Cost (LPZero)')
    plt.title(f'Spearman Correlation: {sp}')
    plt.savefig(f'./output/{structure}_500.png')
    return sp, kd

def sum_arr(arr):
    if isinstance(arr, torch.Tensor):
        return torch.sum(arr).item()
    if isinstance(arr[0], (float, int)):
        return sum(arr)
    
    _sum = 0.0
    for i in range(len(arr)):
        _sum += torch.sum(arr[i])
    return _sum.item() 
    
def fitness_proxy(inputs, proxy_type, device=None, num_samples=500):
    """fitness function for proxy."""
    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('./BERT_results_activation.csv')
    gt_score = []
    zc_score = []

    for i in tqdm(range(num_samples)):
        nas_config = configs[i]['hparams']['model_hparam_overrides']['nas_config']
        gt = configs[i]['scores']['glue']
        # build a new model
        config = ElectraConfig(
            nas_config=nas_config,
            num_hidden_layers=len(nas_config['encoder_layers']),
            output_hidden_states=True,
        )
        model = ElectraModel(config)
        model.to(device)
        if isinstance(inputs, torch.Tensor):
            inputs.to(device)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        # compute zc with the given structure
        if proxy_type == 'fisher':
            zc = compute_fisher_per_weight(model, inputs)
        elif proxy_type == 'snip':
            zc = compute_snip_per_weight(model, inputs)
        elif proxy_type == 'grasp':
            zc = compute_grasp_per_weight(model, inputs)
        elif proxy_type == 'synflow':
            zc = compute_synflow_per_weight(model, inputs)
        elif proxy_type == 'logsynflow':
            zc = compute_logsynflow_per_weight(model, inputs)
        elif proxy_type == 'gradnorm':
            zc = get_grad_norm_arr(model, inputs)
        elif proxy_type == 'act_dist':
            zc = compute_act_dist(model, inputs)
        else: 
            raise NotImplementedError(f'Not support {proxy_type} proxy.')
                
        if isinstance(zc, list):
            zc = zc[0]
            
        zc = sum_arr(zc)        

        if is_anomaly(zc):
            return -1

        # early exit
        if len(zc_score) > 3 and all_same(zc_score):
            return -1

        zc_score.append(zc)
        gt_score.append(gt)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # gc.collect()

    # TODO add inf check
    try:
        if len(zc_score) <= 1 or np.isnan(spearman(gt_score, zc_score)):
            return -1
    except TypeError as e:
        import pdb

        pdb.set_trace()

    df['lpzero'] = zc_score
    df.to_csv('./BERT_results_activation_3.csv', index=False)

    # release memory
    del inputs
    torch.cuda.empty_cache()
    # gc.collect()

    sp = spearman(gt_score, zc_score)
    kd = kendalltau(gt_score, zc_score)
    
    print(f'Spearman of {args.type} is {sp}')
    print(f'Kendalltau of {args.type} is {kd}')
    
if __name__ == '__main__':
    inputs = generate_inputs()
    
    if args.type == 'lpzero':
        # preprocess search space structure
        if args.search_structure == 'linear':
            structure = LinearStructure
        elif args.search_structure == 'tree':
            structure = TreeStructure
        elif args.search_structure == 'graph':
            structure = GraphStructure
        else:
            raise NotImplementedError(
                f'Not support {args.search_structure} structure.')

        print('Begin Evolution Search...')
        struct = structure()
        # genotype = {'input_geno': ['head', 'act'],
        #             'op_geno': [[18, 3], [5, 12], 2]} # 0.7581
        # genotype = {
        #     'input_geno': ['jacobs', 'head'],
        #     'op_geno': [[14, 8], [15, 10], 0],
        # }  # 0.7474
        # genotype = { # better
        #     'input_geno': ['head', 'act'],
        #     'op_geno': [[2, 17], [18, 13], 0],
        # }
        # genotype = { # 0.6928 but better
        #     'input_geno': ['head', 'head'],
        #     'op_geno': [[3, 11], [8, 19], 0],
        # }
        # genotype = { # 0.6803 but better
        #     'input_geno': ['head', 'weight'],
        #     'op_geno': [[8, 19], [17, 8], 0],
        # }
        # genotype = { # 0.6917 but better
        #     'input_geno': ['softmax', 'head'],
        #     'op_geno': [[19, 14], [8, 6], 0],
        # }
        # 2024-02-06 21:08:17 | INFO | evo_search:evolution_search:248 - ====> 
        # After 1000 iters: Best SP:0.8515246098439375 Struct=INPUT:(head, act)TREE:(element_wise_revert|element_wise_pow|frobenius_norm|logsoftmax)BINARY:(element_wise_sum) 
        # Input=['head', 'act'] Op=[[15, 3], [8, 13], 0]

        genotype = {
            'input_geno': ['head', 'act'],
            'op_geno': [[15, 3], [8, 13], 0],
        }

        struct.genotype = genotype
        # Struct=INPUT:(head, act)
        # TREE:(to_std_scalar-element_wise_pow|normalize-sigmoid)
        # BINARY:(element_wise_product) Input=['head', 'act'] Op=[[18, 3], [5, 12], 2]

        sp, kd = fitness_spearman(inputs, struct, num_sample=500)
        print(f'Spearman of {struct} is {sp}')
        print(f'Kendalltau of {struct} is {kd}')

    else:
        # other proxies to be tested 
        fitness_proxy(inputs, proxy_type=args.type)