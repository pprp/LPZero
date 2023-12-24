from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from emq.utils.hessian import (group_product, hessian, hessian_vector_product,
                               normalization)
from . import zc_candidates


@zc_candidates('act')
def compute_activation(net,
                       inputs,
                       targets,
                       loss_fn,
                       split_data=1,
                       **kwargs) -> List:
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
def compute_gradient(net,
                     inputs,
                     targets,
                     loss_fn,
                     split_data=1,
                     **kwargs) -> List:
    grad_list = []  # before relu

    logits = net(inputs)
    loss_fn(logits, targets).backward()

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            grad_list.append(layer.weight.grad.detach())

    return grad_list[::-1]


@zc_candidates('weight')
def compute_weight(net,
                   inputs,
                   targets,
                   loss_fn,
                   split_data=1,
                   **kwargs) -> List:
    weight_list = []

    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            weight_list.append(module.weight.detach())

    _ = net(inputs)
    return weight_list


@zc_candidates('virtual_grad')
def compute_virtual_grad(net,
                         inputs,
                         targets,
                         loss_fn,
                         split_data=1,
                         **kwargs) -> List:
    grad_list = []  # before relu
    input_dim = list(inputs.shape)
    inputs = torch.ones(input_dim, requires_grad=True).to(inputs.device)

    logits = net(inputs)
    torch.sum(logits).backward(retain_graph=True)

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            grad_list.append(layer.weight.grad.detach())

    return grad_list[::-1]


class hessian_per_layer_quant(hessian):
    """ compute the max eigenvalues and trace of hessian in one model by layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_order_grad_dict = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                self.first_order_grad_dict[name] = mod.weight.grad + 0.

    def layer_eigenvalues(self, maxIter=100, tol=1e-3) -> Dict:
        """
        compute the max eigenvalues in one model by layer.
        """
        device = self.device
        max_eigenvalues_dict = {}

        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                weight = mod.weight
                eigenvalue = None
                v = [torch.randn(weight.size()).to(device)]
                v = normalization(v)
                first_order_grad = self.first_order_grad_dict[name]

                for i in range(maxIter):
                    self.model.zero_grad()

                    Hv = hessian_vector_product(first_order_grad, weight, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                    v = normalization(Hv)

                    if eigenvalue is None:
                        eigenvalue = tmp_eigenvalue
                    else:
                        if abs(eigenvalue - tmp_eigenvalue) / (
                                abs(eigenvalue) + 1e-6) < tol:
                            break
                        else:
                            eigenvalue = tmp_eigenvalue
                max_eigenvalues_dict[name] = eigenvalue

        return max_eigenvalues_dict

    def layer_trace(self, maxIter=100, tol=1e-3) -> Dict:
        """
        Compute the trace of hessian in one model by layer.
        """
        device = self.device
        trace_dict = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                trace_vhv = []
                trace = 0.
                weight = mod.weight
                first_order_grad = self.first_order_grad_dict[name]
                for i in range(maxIter):
                    self.model.zero_grad()
                    v = torch.randint_like(weight, high=2, device=device)
                    # generate Rademacher random variables
                    v[v == 0] = -1
                    v = [v]

                    Hv = hessian_vector_product(first_order_grad, weight, v)
                    trace_vhv.append(group_product(Hv, v).cpu().item())
                    if abs(np.mean(trace_vhv) - trace) / (abs(trace) +
                                                          1e-6) < tol:
                        break
                    else:
                        trace = np.mean(trace_vhv)
                trace_dict[name] = trace
        return trace_dict


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