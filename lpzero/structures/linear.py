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


class LinearStructure(BaseStructure):
    """Linear Structure
    Randomly sample one input, and sample `length` unary operations
    to form a linear structure.
    """

    def __init__(self, length=4):
        super().__init__()
        self.length = length
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

    def generate_genotype(self) -> dict:
        """Randomly generate a linear structure."""
        zc_name = self.sample_zc_candidates()
        repr_geno = ''
        repr_geno += f'INPUT:({zc_name})'
        repr_geno += 'UNARY:|'
        op_geno = []
        for _ in range(self.length):
            idx = sample_unary_key_by_prob()
            # random.choice(range(len(UNARY_KEYS)))
            op_geno.append(idx)
            # op_name_list.append(UNARY_KEYS[idx])
            repr_geno += f'{UNARY_KEYS[idx]}|'

        # update _genotype
        self._genotype['input_geno'] = [zc_name]
        self._genotype['op_geno'] = op_geno
        self._repr_geno = repr_geno

    def forward_linear(self, inputs, model):
        """forward to get zc scores"""
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        try:
            A = get_zc_candidates(
                self._genotype['input_geno'][0],
                model,
                device=torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'),
                inputs=img,
                targets=label,
                loss_fn=nn.CrossEntropyLoss(),
            )

            for i in range(len(self._genotype['op_geno'])):
                assert isinstance(A, (list, tuple))
                if len(A) == -1:
                    return -1
                A = [unary_operation(a, self._genotype['op_geno'][i])
                     for a in A]

        except Exception as e:
            print('GOT ERROR in LINEAR STRUCTURE: ', e)
            return -1
        return convert_to_float(A)

    def __call__(self, inputs, model):
        return self.forward_linear(inputs, model)

    def cross_over_by_genotype(self, other):
        """Cross over two genotypes and return new one"""
        assert isinstance(other, LinearStructure)
        assert self.length == other.length

        # cross over input_geno
        idx = random.choice(range(len(self._genotype['input_geno'])))
        input_geno = self._genotype['input_geno']
        input_geno[idx] = other._genotype['input_geno'][idx]

        # cross over op_geno
        idx = random.choice(range(len(self._genotype['op_geno'])))
        op_geno = self._genotype['op_geno']
        op_geno[idx] = other._genotype['op_geno'][idx]

        # cross over repr_geno
        repr_geno = f'INPUT:({input_geno[0]})'
        repr_geno += 'UNARY:|'
        for i in range(len(op_geno)):
            repr_geno += f'{UNARY_KEYS[op_geno[i]]}|'

        geno = {
            'input_geno': input_geno,
            'op_geno': op_geno,
        }

        struct = LinearStructure(self.length)
        struct.genotype = geno
        struct._repr_geno = repr_geno
        return struct

    def mutate_by_genotype(self):
        """Mutate genotype and return linear structure"""
        # mutate input_geno
        idx = random.choice(range(len(self._genotype['input_geno'])))
        input_geno = self._genotype['input_geno']
        input_geno[idx] = self.sample_zc_candidates()

        # mutate op_geno
        idx = random.choice(range(len(self._genotype['op_geno'])))
        op_geno = self._genotype['op_geno']
        op_geno[idx] = random.choice(range(len(UNARY_KEYS)))

        # mutate repr_geno
        repr_geno = f'INPUT:({input_geno[0]})'
        repr_geno += 'UNARY:|'
        for i in range(len(op_geno)):
            repr_geno += f'{UNARY_KEYS[op_geno[i]]}|'

        geno = {
            'input_geno': input_geno,
            'op_geno': op_geno,
        }

        struct = LinearStructure(self.length)
        struct.genotype = geno
        struct._repr_geno = repr_geno
        return struct
