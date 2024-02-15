import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from lpzero.model.flexibert.modeling_electra import ElectraLayer, ElectraModel
from lpzero.model.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex

from . import measure
from ..p_utils import get_layer_metric_array
import torch.nn.functional as F 


def compute_eznas(model, inputs, targets=None, *args, **kwargs):
    grad_output = []
    if isinstance(model, ElectraModel):
        for layer in model.modules():
            if isinstance(layer, ElectraLayer):
                for sublayer in layer.operation.operation.modules():
                    if isinstance(sublayer, nn.Linear):
                        if sublayer.weight.grad is not None:
                            grad_output.append(sublayer.weight.grad)
        output = model(**inputs).last_hidden_state
        output.backward(torch.ones_like(output))
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        assert inputs is not None, "inputs must be provided for computing gradients"
        if targets is None:
            print("current targets is None, maybe invalid for computing gradients")
        loss, _, _, _ = model.forward(inputs, targets, mems=None)
        loss = loss.float().mean().type_as(loss)
        loss.backward()
        
        for layer in model.modules():
            if isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    grad_output.append(layer.weight.grad)
    else: 
        raise NotImplementedError(f"model type {type(model)} is not supported for gradients")
    
    # element-wise-sign, slogdet, sigmoid, frobenius-norm
    def _eznas(_g):
        _g = torch.sign(_g) # sign 
        _g = torch.logdet(_g) # logdet
        _g[_g != _g] = 0
        _g = F.sigmoid(_g) # sigmoid 
        _g = torch.norm(_g, p='fro') # frobenius-norm
        return _g 

    grad_output = [_eznas(g) for g in grad_output]
    

