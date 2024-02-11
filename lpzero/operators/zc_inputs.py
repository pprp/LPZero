from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from lpzero.model.flexibert.modeling_electra import ElectraLayer, ElectraModel
from lpzero.model.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex
from . import zc_candidates
from lpzero.predictor.measures.jacob_cov import modify_net, get_batch_jacobian
import transformers 

@zc_candidates('jacobs')
def compute_jacobs(model, inputs, targets=None, **kwargs) -> List:
    if isinstance(model, ElectraModel):
        output = model(**inputs).last_hidden_state
        output.backward(torch.ones_like(output))
        jacobs = model.embeddings.position_embeddings.weight.grad.detach()
        return jacobs
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        # TODO TO be tested. 
        device = inputs.device
        model = modify_net(model).to(device)
        model.zero_grad()
        jacobs, labels = get_batch_jacobian(
            model, inputs, targets, device, split_data=1
        )
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy() 
        return jacobs 
    else:
        raise NotImplementedError(f"model type {type(model)} is not supported for jacobians")
        


@zc_candidates('act')
def compute_activation(model, inputs, targets=None, **kwargs) -> List:
    """register the activation after each block (for resnet18 the length is 8)"""
    if isinstance(model, ElectraModel):
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
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        act_outputs = []
        def activation_hook(module, input, output):
            act_outputs.append(output.detach())
        for layer in model.modules():
            if isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
                layer.register_forward_hook(activation_hook)
        return act_outputs 
    else: 
        raise NotImplementedError(f"model type {type(model)} is not supported for activations")


@zc_candidates('head')
def compute_head(model, inputs, targets=None, **kwargs) -> List:
    """register the activation after each block (for resnet18 the length is 8)"""
    if isinstance(model, ElectraModel):
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
    
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        head_outputs = []
        def head_hook(module, input, output):
            head_outputs.append(output.detach())
        for layer in model.modules():
            if isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
                layer.register_forward_hook(head_hook)
        return head_outputs


@zc_candidates('softmax')
def compute_softmax(model, inputs, targets=None, **kwargs) -> List:
    """softmax output"""
    if isinstance(model, ElectraModel):
        softmax_outputs = []

        def softmax_hook(module, input, output):
            softmax_outputs.append(output.detach())

        for layer in model.modules():
            if isinstance(layer, ElectraLayer):
                sublayer = layer.operation.operation
                if hasattr(sublayer, 'softmax'):
                    sublayer.softmax.register_forward_hook(softmax_hook)
        model(**inputs)
        return softmax_outputs
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        softmax_outputs = [] 
        
        def softmax_hook(module, input, output):
            softmax_outputs.append(output.detach())
        
        for layer in model.modules():
            # just before the softmax layer
            if isinstance(layer, nn.LayerNorm) and hasattr(layer, 'ln_f'):
                layer.register_forward_hook(softmax_hook)
        
        model.forward(inputs, targets, mems=None)
        return softmax_outputs 
    else: 
        raise NotImplementedError(f"model type {type(model)} is not supported for softmax")

@zc_candidates('grad')
def compute_gradient(model, inputs, targets=None, **kwargs) -> List:
    if isinstance(model, ElectraModel):
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
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        loss, _, _, _ = model.forward(inputs, targets, mems=None)
        loss = loss.float().mean().type_as(loss)
        loss.backward()
        
        grad_output = []
        for layer in model.modules():
            if isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    grad_output.append(layer.weight.grad)
        return grad_output
    else: 
        raise NotImplementedError(f"model type {type(model)} is not supported for gradients")


@zc_candidates('weight')
def compute_weight(model, inputs, targets=None, **kwargs) -> List:
    if isinstance(model, ElectraModel):
        weight_list = []

        for layer in model.modules():
            if isinstance(layer, ElectraLayer):
                for sublayer in layer.operation.operation.modules():
                    if isinstance(sublayer, nn.Linear):
                        if sublayer.weight is not None:
                            weight_list.append(sublayer.weight.detach())
        return weight_list
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        weight_list = []
        for layer in model.modules():
            if isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
                if layer.weight is not None:
                    weight_list.append(layer.weight.detach())
        return weight_list
    else:
        raise NotImplementedError(f"model type {type(model)} is not supported for weights")
