import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class DynamicLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_indices = torch.LongTensor([0, 0])
        out_indices = torch.LongTensor([0, 0])
        self.register_buffer('in_indices', in_indices)
        self.register_buffer('out_indices', out_indices)

    def add_out_indices(self, count):
        self.out_indices[1] += count
        #self.out_indices[1] += self.in_indices[1] - self.in_indices[0] + 1

    def add_in_indices(self, count):
        self.in_indices[1] += count
        #self.in_indices[1] += self.out_indices[1] - self.out_indices[0] + 1

    def set_indices(self, in_indices, out_indices):
        in_indices = torch.LongTensor(in_indices)
        out_indices = torch.LongTensor(out_indices)
        self.in_indices = in_indices
        self.out_indices = out_indices

    def get_in_dimension(self):
        if self.in_indices is None:
            return self.in_features
        else:
            return self.in_indices[1] - self.in_indices[0] + 1

    def get_out_dimension(self):
        if self.out_indices is None:
            return self.out_features
        else:
            return self.out_indices[1] - self.out_indices[0] + 1

    def forward(self, input):
        if self.in_indices is not None:
            w = self.weight[self.out_indices[0]:self.out_indices[1]+1, self.in_indices[0]:self.in_indices[1]+1].contiguous()
            b = self.bias[self.out_indices[0]:self.out_indices[1] + 1].contiguous()
            #logging.info("input.shape={}, self.weight.shape={}, w.shape={}, bias.shape={}",input.shape, self.weight.shape, w.shape, self.bias.shape)
            return F.linear(input, w, b)
        else:
            return super().forward(input)
    
    def build_nn_module(self):
        in_features = self.in_features
        out_features = self.out_features
        if self.in_indices is not None:
            in_features = self.in_indices[1] - self.in_indices[0] + 1
        else:
            in_features = self.in_features
        if self.out_indices is not None:
            out_features = self.out_indices[1] - self.out_indices[0] + 1
        else:
            out_features = self.out_features
        return torch.nn.Linear(in_features, out_features, self.bias is not None)
'''
# FFN   
fc1 = nn.Linear(528, 528*4)  # 528, 528
fc2 = nn.Linear(528*4, 528)

out = fc1(x)  # [B, N, 528*4]
out = fc2(out)


fc1 = DynamicLinear(528, 528*4)
fc2 = DynamicLinear(528*4, 528)
fc1.set_indices([0,527], [0,527])
fc2.set_indices([0,527], [0,527])

out = fc1(x) # [B, N, 528]
out = fc2(out)


# ....训练一个epoch
# 增大block的channel数
fc1.set_indices([0,527], [0,528*2-1])
fc2.set_indices([0,528*2-1], [0,527])

# ....训练一个epoch
# 增大block的channel数
fc1.set_indices([0,527], [0,528*3-1])
fc2.set_indices([0,528*3-1], [0,527])
'''