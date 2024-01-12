import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import ElectraTokenizerFast

from lpzero.model.flexibert.modeling_electra import (
    ElectraConfig,
    ElectraLayer,
    ElectraModel,
)
from lpzero.runner.evo_search import all_same, generate_inputs, is_anomaly, parse_args
from lpzero.structures import GraphStructure, LinearStructure, TreeStructure
from lpzero.utils.rank_consistency import spearman

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
        zc = structure(inputs, model)

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

    # release memory
    del inputs
    torch.cuda.empty_cache()
    # gc.collect()

    sp = spearman(gt_score, zc_score)
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
    return sp


if __name__ == '__main__':
    inputs = generate_inputs()

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
    # } # 0.7474
    # genotype = { # better 
    #     'input_geno': ['head', 'act'],
    #     'op_geno': [[2, 17], [18, 13], 0],
    # } 
    genotype = { # 0.6928 but better 
        'input_geno': ['head', 'head'],
        'op_geno': [[3, 11], [8, 19], 0],
    }
    # genotype = { # 0.6803 but better 
    #     'input_geno': ['head', 'weight'],
    #     'op_geno': [[8, 19], [17, 8], 0],
    # }
    # genotype = { # 0.6917 but better
    #     'input_geno': ['softmax', 'head'],
    #     'op_geno': [[19, 14], [8, 6], 0],
    # }
    struct.genotype = genotype
    # Struct=INPUT:(head, act)
    # TREE:(to_std_scalar-element_wise_pow|normalize-sigmoid)
    # BINARY:(element_wise_product) Input=['head', 'act'] Op=[[18, 3], [5, 12], 2]

    sp = fitness_spearman(inputs, struct, num_sample=500)
    print(f'Spearman of {struct} is {sp}')
