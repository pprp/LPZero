"""This is an implementation of a tree data structure for autoloss."""
import random

import torch
import torch.nn as nn

from lpzero.operators import (
    available_zc_candidates,
    binary_operation,
    get_zc_candidates,
    sample_binary_key_by_prob,
    sample_unary_key_by_prob,
    unary_operation,
)
from lpzero.operators.binary_ops import BINARY_KEYS
from lpzero.operators.unary_ops import UNARY_KEYS
from lpzero.structures.base import BaseStructure
from lpzero.structures.utils import convert_to_float


class TreeStructure(BaseStructure):
    """Tree Structure
    Build Tree-like search space
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
        # regenerate string _repr_geno based on _genotype
        _repr_geno = ''
        _repr_geno += f'INPUT:({self._genotype["input_geno"][0]}, {self._genotype["input_geno"][1]})'
        # for input1
        _repr_geno += f'TREE:({UNARY_KEYS[self._genotype["op_geno"][0][0]]}|{UNARY_KEYS[self._genotype["op_geno"][0][1]]}|'
        # for input2
        _repr_geno += f'{UNARY_KEYS[self._genotype["op_geno"][1][0]]}|{UNARY_KEYS[self._genotype["op_geno"][1][1]]})'
        # for binary operation
        _repr_geno += f'BINARY:({BINARY_KEYS[self._genotype["op_geno"][2]]})'
        return _repr_geno

    @property
    def unique_id(self):
        """Return unique id for this structure"""
        return f'Struct={self} Input={self.genotype["input_geno"]} Op={self.genotype["op_geno"]}'

    def generate_genotype(self):
        """Randomly generate a tree structure.

        For geno:
            input1: [op1, op2]
            input2: [op1, op2]
            binary operation: xx
        """
        zc_name_list = [self.sample_zc_candidates() for _ in range(2)]
        geno = []
        repr_geno = ''
        repr_geno += f'INPUT:({zc_name_list[0]}, {zc_name_list[1]})'
        # for input1
        geno.append([])
        unary1x2 = [sample_unary_key_by_prob() for _ in range(2)]
        geno[0].extend(unary1x2)
        repr_geno += f'TREE:({UNARY_KEYS[unary1x2[0]]}|{UNARY_KEYS[unary1x2[1]]}|'

        # for input2
        geno.append([])
        unary2x2 = [sample_unary_key_by_prob() for _ in range(2)]
        geno[1].extend(unary2x2)
        repr_geno += f'{UNARY_KEYS[unary2x2[0]]}|{UNARY_KEYS[unary2x2[1]]})'

        # for binary operation
        binaryx1 = sample_binary_key_by_prob()
        print('binaryx1', binaryx1)
        repr_geno += f'BINARY:({BINARY_KEYS[binaryx1]})'
        geno.append(binaryx1)

        # update _genotype
        self._genotype['input_geno'] = zc_name_list
        self._genotype['op_geno'] = geno
        self._repr_geno = repr_geno

    def forward_tree(self, inputs, targets, model, return_all=False):
        # try:
        if True:
            A1, A2 = self._genotype['input_geno']
            A1 = get_zc_candidates(
                self._genotype['input_geno'][0],
                model,
                device=torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'),
                inputs=inputs,
                targets=targets,
                loss_fn=nn.CrossEntropyLoss(),
            )
            A2 = get_zc_candidates(
                self._genotype['input_geno'][1],
                model,
                device=torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'),
                inputs=inputs,
                targets=targets,
                loss_fn=nn.CrossEntropyLoss(),
            )

            # process input1 with two unary operations
            A1 = [unary_operation(a, self._genotype['op_geno'][0][0])
                  for a in A1]
            A1 = [unary_operation(a, self._genotype['op_geno'][0][1])
                  for a in A1]

            # process input2 with two unary operations
            A2 = [unary_operation(a, self._genotype['op_geno'][1][0])
                  for a in A2]
            A2 = [unary_operation(a, self._genotype['op_geno'][1][1])
                  for a in A2]

            # process binary operation
            A = []
            for a1, a2 in zip(A1, A2):
                a1 = convert_to_float(a1)
                a2 = convert_to_float(a2)
                A.append(binary_operation(
                    a1, a2, self._genotype['op_geno'][2]))

        # except Exception as e:
        #     print('GOT ERROR in TREE STRUCTURE:', e)
        #     return -1

        if return_all:
            return A, convert_to_float(A)
        else:
            return convert_to_float(A)

    def __call__(self, model, inputs, targets=None, return_all=False):
        return self.forward_tree(inputs=inputs, targets=targets, model=model, return_all=return_all)

    def cross_over_by_genotype(self, other):
        """Cross over two tree structure and return new one"""
        tree = TreeStructure(self.n_nodes)
        # cross over input_geno
        input_geno = []
        for idx in range(2):
            if random.random() < 0.5:
                input_geno.append(self._genotype['input_geno'][idx])
            else:
                input_geno.append(other._genotype['input_geno'][idx])
        tree._genotype['input_geno'] = input_geno

        # cross over op_geno
        op_geno = []
        for idx in range(2):
            op_geno.append([])
            for op_idx in range(2):
                if random.random() < 0.5:
                    op_geno[idx].append(self._genotype['op_geno'][idx][op_idx])
                else:
                    op_geno[idx].append(
                        other._genotype['op_geno'][idx][op_idx])
        tree._genotype['op_geno'] = op_geno

        # cross over binary operation
        if random.random() < 0.5:
            tree._genotype['op_geno'].append(self._genotype['op_geno'][2])
        else:
            tree._genotype['op_geno'].append(other._genotype['op_geno'][2])

        # update repr_geno
        repr_geno = ''
        repr_geno += f'INPUT:({tree._genotype["input_geno"][0]}, {tree._genotype["input_geno"][1]})'
        # for input1
        repr_geno += f'TREE:({UNARY_KEYS[tree._genotype["op_geno"][0][0]]}-{UNARY_KEYS[tree._genotype["op_geno"][0][1]]}|'
        # for input2
        repr_geno += f'{UNARY_KEYS[tree._genotype["op_geno"][1][0]]}-{UNARY_KEYS[tree._genotype["op_geno"][1][1]]})'
        # for binary operation
        repr_geno += f'BINARY:({BINARY_KEYS[tree._genotype["op_geno"][2]]})'
        tree._repr_geno = repr_geno
        return tree

    def mutate_by_genotype(self):
        """return a mutated genotype"""
        tree = TreeStructure(self.n_nodes)
        # mutate input_geno
        input_geno = []
        for idx in range(2):
            if random.random() < 0.5:
                input_geno.append(self._genotype['input_geno'][idx])
            else:
                input_geno.append(random.choice(available_zc_candidates))
        tree._genotype['input_geno'] = input_geno

        # mutate op_geno
        op_geno = []
        for idx in range(2):
            op_geno.append([])
            for op_idx in range(2):
                if random.random() < 0.5:
                    op_geno[idx].append(self._genotype['op_geno'][idx][op_idx])
                else:
                    op_geno[idx].append(sample_unary_key_by_prob())
        tree._genotype['op_geno'] = op_geno

        # mutate binary operation
        if random.random() < 0.5:
            tree._genotype['op_geno'].append(self._genotype['op_geno'][2])
        else:
            tree._genotype['op_geno'].append(sample_binary_key_by_prob())

        # update repr_geno
        repr_geno = ''
        repr_geno += f'INPUT:({tree._genotype["input_geno"][0]}, {tree._genotype["input_geno"][1]})'
        # for input1
        repr_geno += f'TREE:({UNARY_KEYS[tree._genotype["op_geno"][0][0]]}-{UNARY_KEYS[tree._genotype["op_geno"][0][1]]}|'
        # for input2
        repr_geno += f'{UNARY_KEYS[tree._genotype["op_geno"][1][0]]}-{UNARY_KEYS[tree._genotype["op_geno"][1][1]]})'
        # for binary operation
        repr_geno += f'BINARY:({BINARY_KEYS[tree._genotype["op_geno"][2]]})'
        tree._repr_geno = repr_geno
        return tree
