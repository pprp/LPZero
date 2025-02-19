# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Auxiliary methods that dispatches traning/evaluation jobs.
"""

import copy
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import yaml

from lpzero.common import utils
from lpzero.nas_utils.converter import Converter


def _create_batch_jobs(configs: List[Dict[str, Any]],
                       allowed_keys: List[str],
                       default_config_file: str,
                       model_type: str,
                       max_step: int,
                       n_gpus: int,
                       gpu_config: str,
                       dataset: str,
                       vocab: str,
                       vocab_size: int,
                       is_pareto: Optional[bool] = None) -> str:
    """Creates a batch of command-line jobs.

    Args:
        configs: List of configuration dictionaries.
        allowed_keys: Keys that are allowed to be dumped on command-lines.
        default_config_file: Default configuration file.
        model_type: Type of model.
        max_step: Maximum number of training steps.
        n_gpus: Number of GPUs.
        gpu_config: GPU configuration to be used.
        dataset: Name of dataset.
        vocab: Type of vocabulary.
        vocab_size: Size of vocabulary (number of tokens).
        is_pareto: Whether job is from Pareto-frontier or not.

    Returns:
        (str): Command-line command.

    """

    allowed_keys = allowed_keys if allowed_keys is not None else configs[0].keys()
    command = []

    for i, config in enumerate(configs):
        # Maps incoming configuration values to a string-based representation
        n_layer = config['n_layer']
        config_line = ''

        for key, value in config.items():
            if key in allowed_keys:
                if isinstance(value, list):
                    str_value = ' '.join(str(v) for v in value[:n_layer])
                else:
                    str_value = str(value)
                config_line += f' --{key} {str_value}'

        # Checks whether job comes from pareto points, which should have a specific identifier
        is_pareto_str = '_pareto' if (is_pareto is not None and is_pareto[i]) else ''
        exp_name = f'j{i}{is_pareto_str}'

        # Creates the command-line
        if vocab_size is not None:
            line = f'python -m torch.distributed.launch --nproc_per_node={n_gpus} archai/nlp/train.py --model_type {model_type} --config {gpu_config} ' \
                f'--config_file {default_config_file} --dataset {dataset} --max_step {max_step} --vocab {vocab} ' \
                f'--vocab_size {vocab_size} --experiment_name {exp_name} {config_line}'
        else:
            line = f'python -m torch.distributed.launch --nproc_per_node={n_gpus} archai/nlp/train.py --model_type {model_type} --config {gpu_config} ' \
                f'--config_file {default_config_file} --dataset {dataset} --max_step {max_step} --vocab {vocab} ' \
                f'--experiment_name {exp_name} {config_line}'
        
        command.append(line)

    return command


def _create_jobs(configs: List[Dict[str, Any]],
                 allowed_keys: Optional[List[str]] = None,
                 default_config_file: Optional[str] = 'wt103_base.yaml',
                 model_type: Optional[str] = 'mem_transformer',
                 max_step: Optional[int] = 500,
                 start_config: Optional[int] = 0,
                 n_jobs: Optional[int] = 50,
                 n_gpus: Optional[int] = 8,
                 gpu_config: Optional[str] = 'dgx1_8gpu_fp32',
                 dataset: Optional[str] = 'wt103',
                 vocab: Optional[str] = 'gpt2',
                 vocab_size: Optional[int] = None,
                 is_pareto: Optional[bool] = None,
                 output_path: Optional[str] = '~/configs') -> None:
    """Creates command-line jobs.

    Args:
        configs: List of configurations.
        allowed_keys: Keys that are allowed to be dumped on command-lines.
        default_config_file: Default configuration file.
        model_type: Type of model.
        max_step: Number of maximum steps to train the models.
        start_config: Starting range of the configuration to be checked.
        n_jobs: Number of jobs to be created.
        n_gpus: Number of GPUs to be used.
        gpu_config: GPU configuration.
        dataset: Name of dataset.
        vocab: Type of vocabulary.
        vocab_size: Size of vocabulary (number of tokens).
        is_pareto: Whether job is from Pareto-frontier or not.
        output_path: Save folder for the created command-lines.

    """

    output_config_path = utils.full_path(output_path, create=True)
    n_configs = len(configs)
    config_idx = start_config

    c = 0
    while c < n_configs:
        jobs_config = {}
        jobs_config['jobs'] = [{}]

        if is_pareto is not None:
            jobs_config['jobs'][0]['command'] = _create_batch_jobs(copy.deepcopy(configs[c:c+n_jobs]), allowed_keys, default_config_file, model_type, max_step, n_gpus, gpu_config, dataset, vocab, vocab_size, is_pareto=is_pareto[c:c+n_jobs])
        else:
            jobs_config['jobs'][0]['command'] = _create_batch_jobs(copy.deepcopy(configs[c:c+n_jobs]), allowed_keys, default_config_file, model_type, max_step, n_gpus, gpu_config, dataset, vocab, vocab_size)

        output_config_file = os.path.join(output_config_path, f'train_{config_idx}.yaml')
        with open(output_config_file, 'w') as f:
            yaml.dump(jobs_config, f)

        c += n_jobs
        config_idx += 1


def _find_different_genes(population: List[List[Any]],
                          converter: Converter) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Finds different keys that are present in a population.

    Args:
        population: Population to have keys checked.
        converter: Converter class object.

    Returns:
        (Tuple[List[Dict[str, Any]], List[str]]): Non-duplicate configurations and keys.

    """

    configs, keys = [], []

    for gene in population:
        key = converter.gene_to_key(gene)

        if key not in keys:
            configs.append(converter.gene_to_config(gene))
            keys.append(key)

    return configs, keys


def create_pareto_jobs(results_path: str, 
                       converter: Converter,
                       allowed_keys: Optional[List[str]] = None,
                       default_config_file: Optional[str] = 'wt103_base.yaml',
                       model_type: Optional[str] = 'mem_transformer',
                       max_step: Optional[int] = 500,
                       start_config: Optional[int] = 0,
                       n_jobs: Optional[int] = 50,
                       n_gpus: Optional[int] = 8,
                       dataset: Optional[str] = 'wt103',
                       vocab: Optional[str] = 'gpt2',
                       vocab_size: Optional[int] = None, 
                       gpu_config: Optional[str] = 'dgx1_8gpu_fp32',                             
                       output_path: Optional[str] = '~/configs') -> None:
    """Prepares command-line for training Pareto-frontier jobs.

    Args:
        results_path: Path to search results.
        converter: Converter class object.
        allowed_keys: Keys that are allowed to be dumped on command-lines.
        default_config_file: Default configuration file.
        model_type: Type of model.
        max_step: Number of maximum steps to train the models.
        start_config: Starting range of the configuration to be checked.
        n_jobs: Number of jobs to be created.
        n_gpus: Number of GPUs to be used.
        gpu_config: GPU configuration.
        dataset: Name of dataset.
        vocab: Type of vocabulary.
        vocab_size: Size of vocabulary (number of tokens).
        output_path: Save folder for the created command-lines.

    """

    logs_path = os.path.join(results_path, 'logs.pkl')
    with open(logs_path, 'rb') as f:
        logs = pickle.load(f)

    # Gathers pareto population
    # Also checks for different genes because there might be duplicate points in Pareto-frontier
    pareto_population = logs['pareto'][-1]['population']
    pareto_configs, _ = _find_different_genes(pareto_population, converter)

    print(f'Unique Pareto-frontier jobs: {len(pareto_configs)}')

    _create_jobs(pareto_configs,
                 allowed_keys=allowed_keys,
                 default_config_file=default_config_file,
                 model_type=model_type,
                 max_step=max_step,
                 start_config=start_config,
                 n_jobs=n_jobs,
                 n_gpus=n_gpus,
                 gpu_config=gpu_config, 
                 dataset=dataset,
                 vocab=vocab,
                 vocab_size=vocab_size,
                 is_pareto=[True] * len(pareto_configs),
                 output_path=output_path)


def create_ground_truth_jobs(results_path: str,
                             converter: Converter,
                             allowed_keys: Optional[List[str]] = None,
                             default_config_file: Optional[str] = 'wt103_base.yaml',
                             model_type: Optional[str] = 'mem_transformer',
                             max_step: Optional[int] = 500,
                             start_config: Optional[int] = 0,
                             n_jobs: Optional[int] = 50,
                             n_gpus: Optional[int] = 8,
                             gpu_config: Optional[str] = 'dgx1_8gpu_fp32', 
                             dataset: Optional[str] = 'wt103',
                             vocab: Optional[str] = 'gpt2',
                             vocab_size: Optional[int] = None,                            
                             output_path: Optional[str] = '~/configs') -> None:
    """Prepares command-lines for training all visited points during search.

    Args:
        results_path: Path to search results.
        converter: Converter class object.
        allowed_keys: Keys that are allowed to be dumped on command-lines.
        default_config_file: Default configuration file.
        model_type: Type of model.
        max_step: Number of maximum steps to train the models.
        start_config: Starting range of the configuration to be checked.
        n_jobs: Number of jobs to be created.
        n_gpus: Number of GPUs to be used.
        gpu_config: GPU configuration.
        dataset: Name of dataset.
        vocab: Type of vocabulary.
        vocab_size: Size of vocabulary (number of tokens).
        output_path: Save folder for the created command-lines.

    """

    logs_path = os.path.join(results_path, 'logs.pkl')
    with open(logs_path, 'rb') as f:
        logs = pickle.load(f)

    # Gathers total populations and pareto population
    total_population = logs['population']
    pareto_population = logs['pareto'][-1]['population']

    # Checks for different genes because there might be duplicate points in Pareto-frontier
    _, pareto_keys = _find_different_genes(pareto_population, converter)

    # Iterates through all populations and gathers a list of non-duplicated configurations
    total_population_configs, total_population_keys = [], []
    for population in total_population:
        population_configs, population_keys = _find_different_genes(population, converter)

        for p_config, p_key in zip(population_configs, population_keys):
            if p_key not in total_population_keys and p_key not in pareto_keys:
                total_population_configs.append(p_config)
                total_population_keys.append(p_key)

    print(f'Unique jobs (non-Pareto): {len(total_population_configs)}')

    _create_jobs(total_population_configs,
                 allowed_keys=allowed_keys,
                 default_config_file=default_config_file,
                 model_type=model_type,
                 max_step=max_step,
                 start_config=start_config,
                 n_jobs=n_jobs,
                 n_gpus=n_gpus,
                 gpu_config=gpu_config, 
                 dataset=dataset,
                 vocab=vocab,
                 vocab_size=vocab_size,
                 output_path=output_path)
