import argparse
import csv
import json
import os 

import math
import random
from typing import Union
import collections
import sys  
import pdb
import traceback 

import re 
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from torch import Tensor


from lpzero.structures import GraphStructure, LinearStructure, TreeStructure
from lpzero.utils.rank_consistency import spearman
import numpy as np
from scipy.stats import spearmanr
from lpzero.model.model_loader import load_model_from_config

from lpzero.utils.rank_consistency import kendalltau
from lpzero.datasets.distributed_utils.data_utils import get_lm_corpus

# save memory to file    
mem_dict = {}

def get_batch_data(train_iter, num_batches, device):
    traindata = []
    # wkitext103
    train_iter = train_iter.get_fixlen_iter(start=0)
        
    for batch, (data, target, _, _) in enumerate(train_iter, start=1):
        if batch > num_batches:
            break
        traindata.append((data, target))

    inputs = torch.cat([a for a, _ in traindata], dim=-1)
    targets = torch.cat([b for _, b in traindata], dim=-1)
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets 


def parse_args():
    parser = argparse.ArgumentParser(
        description='running parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--step', default=20, type=int, help='record snn output per step'
    )

    parser.add_argument(
        '--iterations', default=1000, type=int, help='number of iteration for LSQ'
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
        default=80,
        type=int,
        help='population size should be larger than 10',
    )

    # log path
    parser.add_argument(
        '--log_path',
        default='./logs/evo_search_run0.log',
        type=str,
        help='path of log',
    )
    # num_sample
    parser.add_argument(
        '--num_sample',
        default=50,
        type=int,
        help='number of sample to be evaluate the ranking consistency',
    )
    parser.add_argument( "--exp_name", type=str, default=".", 
                        help="path to the experiment",)
    parser.add_argument('--model_type', default='hf_gpt2_flex', type=str,
                     choices=['hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl', 'mem_transformer'],
                     help='Which model type to use')
    parser.add_argument("--get_cost", action="store_true", help="compute cost for each method")
    parser.add_argument("--seed", type=int, default=1111, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="plot the spearman corr and common ratio")
    parser.add_argument("--method", type=str, default="snip", help="zero-cost method to use")
    parser.add_argument("--cuda", action="store_true", help="use gpu for score calculation")

    parser.add_argument("--batch_size", type=int, default=16, help="Global batch size")
    parser.add_argument("--dataset", type=str, default="wt103", choices=["wt103", "lm1b"], help="Dataset name",)
    parser.add_argument("--vocab", type=str, default="gpt2", choices=["gpt2"], help="Type of vocabulary")
    parser.add_argument("--vocab_size", type=int, default=None, help="Size of vocabulary")
    parser.add_argument("--dataload", type=str, default="random", help="random or grasp supported")
    parser.add_argument("--dataload_info", type=int, default=1, 
                        help="number of batches to use for random dataload or number of samples per class for grasp dataload",)
    parser.add_argument("--start", type=int, default=5, help="start index")
    parser.add_argument("--end", type=int, default=10, help="end index")
    parser.add_argument("--write_freq", type=int, default=100, help="frequency of write to file")
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


args = parse_args()

print(f"LOG SAVE PATH: {args.log_path}")

logger.add(
    args.log_path,
    format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}',
    level='INFO',
)


def all_same(items):
    return all(x == items[0] for x in items)


def convert_path_to_config_format(path: str) -> str:
    # Split the path into parts
    parts = path.split('/')
    
    # Extract the relevant parts (config_xx and jy)
    config_part = parts[-3]  # This gets 'config_xx'
    j_part = parts[-2]      # This gets 'jy'
    
    # Combine them into the desired format
    formatted_string = f"{config_part}_{j_part}"
    
    return formatted_string


def get_scores(structure, tr_iter=None):
    assert structure is not None, "structure is not defined"
    path_to_results = './saved_logs/random_GPT2_wt103'
    device = torch.device("cuda" if args.cuda else "cpu")
    inputs, targets = get_batch_data(tr_iter, 1, device)
    
    files = []
    dirlist = [path_to_results]
    while len(dirlist) > 0:
        for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
            dirlist.extend([os.path.join(dirpath, d) for d in dirnames])
            files.extend(map(lambda n: os.path.join(*n), zip([dirpath] * len(filenames), filenames),))
    
    # load the ground-truth rankings
    yaml_file = os.path.join(path_to_results, "ppl_summary.yaml")
    with open(yaml_file, "r") as f:
        results_gt = collections.OrderedDict(yaml.safe_load(f))
    
    gt_list = []
    zc_list = []
        
    # almost 200 architectures
    # random sample 50 architectures to evaluate the ranking consistency
    files = random.sample(files, args.num_sample)
    
    for _f in set(files):
        if "model_config.yaml" in _f:
            idx =  re.search('(config_[0-9]+)', _f).span()[0]
            with open(_f, "r") as f:
                model_config = yaml.full_load(f)
            
            model = load_model_from_config(args.model_type, model_config)
            model.n_token = model_config["n_token"]
            model.to(device)
                        
            # replace to structure 
            zc = structure(model, inputs, targets)
            gt_list.append(results_gt[convert_path_to_config_format(_f)]["valid_ppl"])
            zc_list.append(zc)
            
            del model 
            import gc 
            gc.collect()
            torch.cuda.empty_cache()
    
    sp = spearman(gt_list, zc_list)
    kd = kendalltau(gt_list, zc_list)
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


def evolution_search(structure, train_itr, iterations=1000, popu_size=50):
    # random initialize N structures for evolution
    population = []
    logger.info('Initialize population')

    while len(population) < popu_size:
        struct = structure()
        score = get_scores(struct, train_itr)
        if is_anomaly(score):
            continue
        logger.info(f'Current population size: {len(population)}')
        logger.info(f'Current structure: {struct} with score: {score}')
        population.append(struct)

    # prepare data for matplotlib plot
    idx = []
    sps = []

    # run the cycle
    logger.info('Begin the evolution process...')
    for i in range(iterations):
        scores = [get_scores(struct, train_itr) for struct in population]
        # select the best one from the population
        scores = np.array(scores)
        argidxs = np.argsort(scores)[::-1]

        # best structure on the run
        running_struct = population[argidxs[0]]
        logger.info(
            f'===> Iter: {i} Best SP: {scores[argidxs[0]]} Best Struct={running_struct} Input={running_struct.genotype["input_geno"]} Op={running_struct.genotype["op_geno"]}'
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

        del population[argidxs[-1]]

        population.append(offspring_struct)
        score = get_scores(offspring_struct, train_itr)
        logger.info(
            f'Iter: {i} Offspring SP: {score} Offspring Struct={offspring_struct} Input={offspring_struct.genotype["input_geno"]} Op={offspring_struct.genotype["op_geno"]}'
        )

        # 6. assert the population size should not shrink
        assert len(
            population) == popu_size, f'Population size should be {popu_size}'

    # evaluate the fitness of all structures
    scores = [get_scores(s, train_itr) for s in population]
    argidxs = np.argsort(scores)[::-1]
    running_struct = population[argidxs[0]]
    logger.info(
        f'====> After {iterations} iters: Best SP:{scores[argidxs[0]]} Struct={running_struct} Input={running_struct.genotype["input_geno"]} Op={running_struct.genotype["op_geno"]}'
    )

    # plot the evolution process
    save_name = f'evolution_{type(offspring_struct)}_{iterations}_{popu_size}_{random.randint(0, 1000)}_test'
    plt.plot(idx, sps)
    plt.xlabel('Iteration')
    plt.ylabel('Spearman')
    if not os.path.exists('./output/evo_search_emq_zc'):
        os.makedirs('./output/evo_search_emq_zc')

    plt.savefig(f'./output/evo_search_emq_zc/{save_name}.png')

    # save idx and sps into csv file
    if not os.path.exists('./output/csv_files'):
        os.makedirs('./output/csv_files')
    with open(f'./output/csv_files/{save_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(idx, sps))

    return running_struct


def main():
    if args.dataset == "wt103":
        eval_batch_size = 16
        eval_tgt_len = 192
    elif args.dataset == "lm1b":
        eval_batch_size = 16
        eval_tgt_len = 32
        
    device = torch.device("cuda" if args.cuda else "cpu")

    data = './data/wikitext/wikitext-103'
    cache_dir = './data/cachedir'
    vocab = 'gpt2' if 'gpt' in args.model_type else 'word'
    vocab_size = 50264 if 'gpt' in args.model_type else 267736
    corpus = get_lm_corpus(data, cache_dir, args.dataset, vocab, vocab_size, refresh_cache=False)
    train_itr = corpus.get_iterator("train", eval_batch_size, eval_tgt_len,
                                    device=device, mem_len=0, ext_len=0)
    args.n_token = len(corpus.vocab)

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
    evolution_search(structure, train_itr, args.iterations, args.popu_size)

    # save the memory to file
    if not os.path.exists(f'./output/memory'):
        os.makedirs(f'./output/memory')

    exp_name = os.path.basename(args.log_path).split('.')[0]
    with open(f'./output/memory/mem_dict_{exp_name}.json', 'w') as f:
        json.dump(mem_dict, f)
    
if __name__ == "__main__":
    try: 
        main()
    except: 
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)