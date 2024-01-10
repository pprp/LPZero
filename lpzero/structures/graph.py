"""This is an implementation of a graph data structure for autoloss."""
import math
import random

import torch
import torch.nn as nn

from lpzero.operators import (
    available_zc_candidates,
    get_zc_candidates,
    sample_unary_key_by_prob,
    unary_operation,
)
from lpzero.operators.unary_ops import UNARY_KEYS
from lpzero.structures.base import BaseStructure
from lpzero.structures.utils import convert_to_float


class GraphStructure(BaseStructure):
    """Graph Structure
    Build a DAG(Directed Acyclic Graph) structure
    """

    def __init__(self, n_nodes=3):
        super().__init__()
        self.n_nodes = n_nodes
        self._sp_score = -1  # -1 denotes invalid
        self._genotype = {
            'input_geno': [],  # only one
            'op_geno': [],  # length
        }
        # init _genotype
        self.generate_genotype()

    def sample_zc_candidates(self) -> str:
        """sample one input from zc candidates"""
        total_num_zc_candidates = len(available_zc_candidates)
        idx_zc = random.choice(range(total_num_zc_candidates))
        return available_zc_candidates[idx_zc]
    
    def __str__(self):
        _repr_geno = ''
        _repr_geno += f'INPUT:({self._genotype["input_geno"][0]}, {self._genotype["input_geno"][1]})UNARY:('
        for i in range(self.n_nodes):
            for j in range(i + 2):
                _repr_geno += UNARY_KEYS[self._genotype["op_geno"][i][j]] + '|'
            _repr_geno += '-> '
        _repr_geno += ')'
        return _repr_geno

    def generate_genotype(self):
        """Randomly generate a graph structure."""
        zc_name_list = [self.sample_zc_candidates() for _ in range(2)]
        dag = []
        repr_geno = ''
        repr_geno += f'INPUT:({zc_name_list[0]}, {zc_name_list[1]})UNARY:('
        for i in range(self.n_nodes):
            dag.append([])
            for j in range(i + 2):  # include 2 input nodes
                # sample unary operation
                idx = sample_unary_key_by_prob()
                # random.choice(range(len(UNARY_KEYS)))
                dag[i].append(idx)
                repr_geno += UNARY_KEYS[idx] + '|'
            repr_geno += '-> '
        repr_geno += ')'

        # update _genotype
        self._genotype['input_geno'] = zc_name_list
        self._genotype['op_geno'] = dag
        self._repr_geno = repr_geno

    def forward_dag(self, inputs, model):
        """forward DAG and return ZC score"""
        
        states = []
        for zc_name in self._genotype['input_geno']:
            states.append(get_zc_candidates(zc_name, 
                          model=model, 
                          device=torch.device(
                              'cuda' if torch.cuda.is_available() else 'cpu'),
                          inputs=inputs,
                          loss_fn=nn.CrossEntropyLoss()))
        # try:
        if True:
            for edges in self._genotype['op_geno']:
                assert len(states) == len(
                    edges
                ), f'length of states should be {len(edges)}'
                # states[i] is list of tensor
                midst = []
                for idx, state in zip(edges, states):
                    tmp = []
                    for s in state:
                        tmp.append(unary_operation(s, idx))
                    midst.append(tmp)

                # merge N lists of tensor to one list of tensor
                res = []
                for i in range(len(midst[0])):
                    t = midst[0][i]
                    for j in range(1, len(midst)):
                        t += midst[j][i]
                    res.append(t)
                states.append(res)

            res_list = []
            for item in states[2:]:
                res = convert_to_float(item)
                if math.isnan(res) or math.isinf(res):
                    return -1  # invalid
                res_list.append(res)
        # except Exception as e:
        #     print('GOT ERROR in GRAPH STRUCTURE: ', e)
        #     return -1  # invalid

        # check whether the res_list of float have inf,nan
        return sum(res_list) / len(res_list)

    def __call__(self, inputs, model):
        return self.forward_dag(inputs, model)

    def cross_over_by_genotype(self, other):
        """cross over two graph structures and return new genotype"""
        assert isinstance(other, GraphStructure), 'type error'
        # crossover input_geno
        input_geno = [self._genotype['input_geno']
                      [0], other._genotype['input_geno'][1]]

        # crossover op_geno
        op_geno = []
        for i in range(self.n_nodes):
            op_geno.append([])
            for j in range(i + 2):
                if random.choice([True, False]):
                    op_geno[i].append(self._genotype['op_geno'][i][j])
                else:
                    op_geno[i].append(other._genotype['op_geno'][i][j])

        # rephrase repr_geno
        repr_geno = ''
        repr_geno += f'INPUT:({input_geno[0]}, {input_geno[1]})UNARY:('
        for i in range(self.n_nodes):
            for j in range(i + 2):
                repr_geno += UNARY_KEYS[op_geno[i][j]] + '|'
            repr_geno += '-> '
        repr_geno += ')'
        geno = {
            'input_geno': input_geno,
            'op_geno': op_geno,
        }
        struct = GraphStructure(self.n_nodes)
        struct.genotype = geno
        struct._repr_geno = repr_geno
        return struct

    def mutate_by_genotype(self):
        """return to new genotype"""
        # input_geno
        input_geno = [self._genotype['input_geno']
                      [0], self.sample_zc_candidates()]

        # op_geno
        op_geno = []
        for i in range(self.n_nodes):
            op_geno.append([])
            for j in range(i + 2):
                if random.choice([True, False]):
                    op_geno[i].append(random.choice(range(len(UNARY_KEYS))))
                else:
                    op_geno[i].append(self._genotype['op_geno'][i][j])

        # repr_geno
        repr_geno = ''
        repr_geno += f'INPUT:({input_geno[0]}, {input_geno[1]})UNARY:('
        for i in range(self.n_nodes):
            for j in range(i + 2):
                repr_geno += f'{UNARY_KEYS[op_geno[i][j]]}|'
            repr_geno += '-> '
        repr_geno += ')'
        geno = {
            'input_geno': input_geno,
            'op_geno': op_geno,
        }

        struct = GraphStructure(self.n_nodes)
        struct.genotype = geno
        struct._repr_geno = repr_geno
        return struct
