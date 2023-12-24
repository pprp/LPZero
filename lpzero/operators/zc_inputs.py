from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from . import zc_candidates


@zc_candidates('act')
def compute_activation(net, inputs, targets, loss_fn, split_data=1, **kwargs) -> List:
    """register the activation after each block (for resnet18 the length is 8)"""

    act_list = []

    def hook_fw_act_fn(module, input, output):
        act_list.append(output.detach())

    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            module.register_forward_hook(hook_fw_act_fn)

    _ = net(inputs)
    return act_list


@zc_candidates('grad')
def compute_gradient(net, inputs, targets, loss_fn, split_data=1, **kwargs) -> List:
    grad_list = []  # before relu

    logits = net(inputs)
    loss_fn(logits, targets).backward()

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            grad_list.append(layer.weight.grad.detach())

    return grad_list[::-1]


@zc_candidates('weight')
def compute_weight(net, inputs, targets, loss_fn, split_data=1, **kwargs) -> List:
    weight_list = []

    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            weight_list.append(module.weight.detach())

    _ = net(inputs)
    return weight_list


# @zc_candidates('hessian_eigen')
# def compute_hessian_eigen(net,
#                           inputs,
#                           targets,
#                           loss_fn,
#                           split_data=1,
#                           **kwargs) -> List:
#     cuda = True if torch.cuda.is_available() else False
#     hessian_comp = hessian_per_layer_quant(
#         model=net, criterion=loss_fn, data=(inputs, targets), cuda=cuda)

#     eigens = hessian_comp.layer_eigenvalues()

#     res = []
#     for v in eigens.values():
#         res.append(float(v))
#     return res

# @zc_candidates('hessian_trace')
# def compute_hessian_trace(net,
#                           inputs,
#                           targets,
#                           loss_fn,
#                           split_data=1,
#                           **kwargs) -> List:
#     cuda = True if torch.cuda.is_available() else False
#     hessian_comp = hessian_per_layer_quant(
#         model=net, criterion=loss_fn, data=(inputs, targets), cuda=cuda)

#     traces = hessian_comp.layer_trace()

#     res = []
#     for v in traces.values():
#         res.append(float(v))
#     return res
