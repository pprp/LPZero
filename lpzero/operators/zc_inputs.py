from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from lpzero.model.flexibert.modeling_electra import ElectraLayer
from . import zc_candidates


@zc_candidates('jacobs')
def compute_jacobs(model, inputs, **kwargs) -> List:
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    jacobs = model.embeddings.position_embeddings.weight.grad.detach()
    return jacobs


@zc_candidates('act')
def compute_activation(model, inputs, **kwargs) -> List:
    """register the activation after each block (for resnet18 the length is 8)"""
    act_outputs = []

    def activation_hook(module, input, output):
        act_outputs.append(output.detach())

    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            sublayer = layer.intermediate.intermediate_act_fn.register_forward_hook(
                activation_hook
            )
    model(**inputs)
    return act_outputs


@zc_candidates('head')
def compute_head(model, inputs, **kwargs) -> List:
    """register the activation after each block (for resnet18 the length is 8)"""
    head_outputs = []

    def head_hook(module, input, output):
        head_outputs.append(output.detach())

    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            sublayer = layer.operation.operation
            if hasattr(sublayer, 'query'):
                sublayer.query.register_forward_hook(head_hook)
            elif hasattr(sublayer, 'key'):
                sublayer.key.register_forward_hook(head_hook)
            elif hasattr(sublayer, 'value'):
                sublayer.value.register_forward_hook(head_hook)
            elif hasattr(sublayer, 'input'):
                sublayer.input.register_forward_hook(head_hook)
            elif hasattr(sublayer, 'weight'):
                sublayer.weight.register_forward_hook(head_hook)

    model(**inputs)
    return head_outputs


# @zc_candidates('softmax')
# def compute_softmax(model, inputs, **kwargs) -> List:
#     """softmax output"""
#     softmax_outputs = []

#     def softmax_hook(module, input, output):
#         softmax_outputs.append(output.detach())

#     for layer in model.modules():
#         if isinstance(layer, ElectraLayer):
#             sublayer = layer.operation.operation
#             if hasattr(sublayer, 'softmax'):
#                 sublayer.softmax.register_forward_hook(softmax_hook)
#     model(**inputs)
#     return softmax_outputs


@zc_candidates('grad')
def compute_gradient(model, inputs, **kwargs) -> List:
    grad_output = []
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            for sublayer in layer.operation.operation.modules():
                if isinstance(sublayer, nn.Linear):
                    if sublayer.weight.grad is not None:
                        grad_output.append(sublayer.weight.grad)
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    return grad_output


@zc_candidates('weight')
def compute_weight(model, inputs, **kwargs) -> List:
    weight_list = []

    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            for sublayer in layer.operation.operation.modules():
                if isinstance(sublayer, nn.Linear):
                    if sublayer.weight is not None:
                        weight_list.append(sublayer.weight.detach())
    return weight_list
