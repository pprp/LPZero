import torch
from . import measure

from lpzero.model.flexibert.modeling_electra import ElectraLayer, ElectraModel
from lpzero.model.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex

import transformers 

# Attention Head Importance metric
def head_importance(model):
    if isinstance(model, ElectraModel):
        metric_array = []
        for layer in model.modules():
            if isinstance(layer, ElectraLayer):
                for sublayer in layer.operation.operation.modules():
                    if isinstance(sublayer, torch.nn.Linear):
                        if (
                            (sublayer.weight is not None)
                            and (sublayer.weight.grad is not None)
                            and sublayer.weight.shape[0] >= 128
                        ):
                            metric_array.append(
                                torch.abs(sublayer.weight.data *
                                        sublayer.weight.grad)
                            )
        summed = torch.tensor(0.0).to('cuda')
        for j in range(len(metric_array)):
            summed += torch.nansum(metric_array[j])
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        metric_array = []
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Linear, transformers.Conv1D)):
                if (
                    (layer.weight is not None)
                    and (layer.weight.grad is not None)
                    and layer.weight.shape[0] >= 128
                ):
                    metric_array.append(
                        torch.abs(layer.weight.data * layer.weight.grad)
                    )
        summed = torch.tensor(0.0).to('cuda')
        for j in range(len(metric_array)):
            summed += torch.nansum(metric_array[j])

    return summed.detach().item()


def head_importance_normalized(model):
    metric_array = []
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            for sublayer in layer.operation.operation.modules():
                if isinstance(sublayer, torch.nn.Linear):
                    if (
                        (sublayer.weight is not None)
                        and (sublayer.weight.grad is not None)
                        and sublayer.weight.shape[0] >= 128
                    ):
                        metric_array.append(
                            torch.abs(sublayer.weight.data *
                                      sublayer.weight.grad)
                        )
    summed = torch.tensor(0.0).to('cuda')
    for j in range(len(metric_array)):
        summed += torch.nansum(metric_array[j])
    summed /= len(metric_array)

    return summed.detach().item()

@measure('head_importance')
def compute_head_importance(model, inputs, targets=None, *args, **kwargs):
    if isinstance(model, ElectraModel):
        model.zero_grad()
        outputs = model(**inputs).last_hidden_state
        outputs.backward(torch.ones_like(outputs))
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        loss, _, _, _ = model.forward(inputs, targets, mems=None)
        loss = loss.float().mean().type_as(loss)
        loss.backward()
    return head_importance(model)