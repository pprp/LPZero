# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import torch.nn as nn
import torch.autograd as autograd
import transformers

from lpzero.model.flexibert.modeling_electra import ElectraModel
from lpzero.model.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex

from . import measure
from ..p_utils import get_layer_metric_array


@measure("grasp", bn=True, mode="param", copy_net=False)
def compute_grasp_per_weight(
    net, inputs, targets=None, mode='param'):
    
    # get all applicable weights
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
            layer.weight.requires_grad_(True)  # TODO isn't this already true?
            layer.compute = 0

    # NOTE original code had some input/target splitting into 2
    # I am guessing this was because of GPU mem limit
    net.zero_grad()

    # forward/grad pass #1
    grad_w = None
    if isinstance(net, (HfGPT2, HfGPT2Flex)):
        loss, _, _, _ = net.forward(inputs, targets, mems=None)
        loss = loss.float().mean().type_as(loss)
    elif isinstance(net, ElectraModel):
        outputs = net(**inputs).last_hidden_state 
        loss = outputs.sum()
    
    grad_w_p = autograd.grad(loss, weights, allow_unused=True)
    
    if grad_w is None:
        grad_w = list(grad_w_p)
    else:
        for idx in range(len(grad_w)):
            grad_w[idx] += grad_w_p[idx]

    # forward/grad pass #2
    if isinstance(net, (HfGPT2, HfGPT2Flex)):
        loss, _, _, _ = net.forward(inputs, targets, mems=None)
        loss = loss.float().mean().type_as(loss)
    elif isinstance(net, ElectraModel):
        outputs = net(**inputs).last_hidden_state 
        loss = outputs.sum()
    
    grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)

    # accumulate gradients computed in previous step and call backwards
    z, count = 0, 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
            if grad_w[count] is not None:
                z += (grad_w[count].data * grad_f[count]).sum()
                layer.compute += torch.prod(torch.tensor(grad_w[count].size())).item()
            count += 1
    z.backward()

    # compute final sensitivity metric and put in grads
    def grasp(layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad  # -theta_q Hg
            # NOTE in the grasp code they take the *bottom* (1-p)% of values
            # but we take the *top* (1-p)%, therefore we remove the -ve sign
            # EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
        else:
            return torch.zeros_like(layer.weight)

    grads = get_layer_metric_array(net, grasp, mode)

    return grads
