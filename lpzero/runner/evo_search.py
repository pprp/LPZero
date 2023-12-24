from loguru import logger
import argparse
import csv
import json

# build logger
import logging
import math
import random
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch import Tensor

from lpzero.model.flexibert.modeling_electra import (
    ElectraConfig,
    ElectraLayer,
    ElectraModel,
)
from lpzero.structures import GraphStructure, LinearStructure, TreeStructure
from lpzero.utils.rank_consistency import spearman
from transformers import ElectraTokenizerFast

configs = []
with open('./data/BERT_benchmark.json', 'r') as f:
    configs = json.load(f)


logger.add(
    'info.log',
    format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}',
    level='INFO',
)


def all_same(items):
    return all(x == items[0] for x in items)


def fitness_spearman(dataiter, structure, inputs, device, num_sample=50):
    """structure is belong to popultion."""
    if structure.sp_score != -1:
        return structure.sp_score

    gt_score = []
    zc_score = []

    for i in range(num_sample):
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
    return sp


def is_anomaly(zc_score: Union[torch.Tensor, float, int] = None) -> bool:
    """filter the score with -1,0,nan,inf"""
    if isinstance(zc_score, Tensor):
        zc_score = zc_score.item()

    if (
        zc_score is None
        or zc_score == -1
        or math.isnan(zc_score)
        or math.isinf(zc_score)
        or zc_score == 0
    ):
        return True
    return False


def evolution_search(dataiter, structure, iterations=1000, popu_size=50):
    # random initialize N structures for evolution
    population = []
    logger.info('Initialize population')

    while len(population) < popu_size:
        struct = structure()
        score = fitness_spearman(dataiter, struct)
        if is_anomaly(score):
            continue
        logger.info(f'Current population size: {len(population)}')
        population.append(struct)

    # prepare data for matplotlib plot
    idx = []
    sps = []

    # run the cycle
    logger.info('Begin the evolution process...')
    for i in range(iterations):
        scores = [fitness_spearman(dataiter, struct) for struct in population]
        # select the best one from the population
        scores = np.array(scores)
        argidxs = np.argsort(scores)[::-1]

        # best structure on the run
        running_struct = population[argidxs[0]]
        logger.info(
            f'Iter: {i} Best KD: {scores[argidxs[0]]} Struct={running_struct} Input={running_struct.genotype["input_geno"]} Op={running_struct.genotype["op_geno"]}'
        )

        # add data for matplotlib plot
        idx.append(i)
        sps.append(scores[argidxs[0]])

        candidates = [population[i]
                      for i in argidxs[: int(popu_size * 0.5)]]  # TopN
        offspring_struct = random.choice(candidates)
        best_struct2 = random.choice(candidates)

        if np.random.rand() < 0.5:
            # 1. cross over
            offspring_struct = offspring_struct.cross_over_by_genotype(
                best_struct2)

        if np.random.rand() < 0.5:
            # 2. mutation
            offspring_struct = offspring_struct.mutate_by_genotype()

        # 3. Diversity-prompting selection
        # offspring_zc = fitness_spearman(dataiter,  offspring_struct)
        # newbie = structure()
        # newbie_zc = fitness_spearman(dataiter,  newbie)

        # 4. delete the deteriorated structure
        del population[argidxs[-1]]

        # 5. add better offspring or newbie to population
        # if offspring_zc > newbie_zc:
        #     population.append(offspring_struct)
        # else:
        #     population.append(newbie)
        population.append(offspring_struct)

        # 6. assert the population size should not shrink
        assert len(
            population) == popu_size, f'Population size should be {popu_size}'

    # evaluate the fitness of all structures
    scores = [fitness_spearman(dataiter, s) for s in population]
    argidxs = np.argsort(scores)[::-1]
    running_struct = population[argidxs[0]]
    logger.info(
        f'After {iterations} iters: Best KD:{scores[argidxs[0]]} Struct={running_struct} Input={running_struct.genotype["input_geno"]} Op={running_struct.genotype["op_geno"]}'
    )

    # plot the evolution process
    save_name = f'evolution_{type(newbie)}_{iterations}_{popu_size}_{random.randint(0, 1000)}_DPS'
    plt.plot(idx, sps)
    plt.xlabel('Iteration')
    plt.ylabel('Spearman')
    plt.savefig(f'./output/evo_search_emq_zc/{save_name}.png')

    # save idx and sps into csv file
    with open(f'./output/evo_search_emq_zc/{save_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(idx, sps))

    return running_struct


def generate_inputs():
    # Target dataset is openwebtex
    dataset = load_dataset('openwebtext')
    tokenizer = ElectraTokenizerFast.from_pretrained(
        'google/electra-small-discriminator'
    )

    def encode(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    tokenized_dataset = dataset.map(encode, batched=True, num_proc=32)
    tokenized_dataset.set_format(
        type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask']
    )

    # get sample tokenized batch from dataset
    dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=128)
    inputs = tokenizer(
        next(iter(dataloader))['text'],
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    return inputs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='running parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # general parameters for data and qnn
    parser.add_argument(
        '--seed', default=42, type=int, help='random seed for results reproduction'
    )
    parser.add_argument(
        '--step', default=20, type=int, help='record snn output per step'
    )

    parser.add_argument(
        '--iterations', default=1, type=int, help='number of iteration for LSQ'
    )
    # search space structure
    parser.add_argument(
        '--search_structure',
        default='tree',
        type=str,
        help='search space structure, linear, tree or graph',
    )

    # popu size
    parser.add_argument(
        '--popu_size',
        default=11,
        type=int,
        help='population size should be larger than 10',
    )

    args = parser.parse_args()

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

    logger.info('Begin Evolution Search...')
    evolution_search(args.dataiter, structure, args.iterations, args.popu_size)
