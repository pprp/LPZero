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

def head_confidence(outputs):
    metric_array = []
    for output in outputs:
        metric_array.append(torch.mean(torch.max(output, 1)[0]))

    summed = torch.tensor(0.0).to('cuda')
    for j in range(len(outputs)):
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

@measure('head_confidence')
def compute_head_confidence(model, inputs, targets=None, *args, **kwargs):
    head_outputs = []
    if isinstance(model, ElectraModel):
        def head_hook(module, input, output):
            head_outputs.append(output)

        # Initialize hooks
        for layer in model.modules():
            if isinstance(layer, ElectraLayer):
                sublayer = layer.operation.operation
                if hasattr(sublayer, 'query'):
                    sublayer.query.register_forward_hook(head_hook)
                if hasattr(sublayer, 'key'):
                    sublayer.key.register_forward_hook(head_hook)
                if hasattr(sublayer, 'value'):
                    sublayer.value.register_forward_hook(head_hook)
                if hasattr(sublayer, 'input'):
                    sublayer.input.register_forward_hook(head_hook)
                if hasattr(sublayer, 'weight'):
                    sublayer.weight.register_forward_hook(head_hook)
        model.zero_grad()
        output = model(**inputs).last_hidden_state
        output.backward(torch.ones_like(output))
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        def head_hook(module, input, output):
            head_outputs.append(output)

        # Initialize hooks
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear) or \
                isinstance(layer, transformers.Conv1D):
                layer.register_forward_hook(head_hook)
        loss, _, _, _ = model.forward(inputs, targets, mems=None)
        loss = loss.float().mean().type_as(loss)
        loss.backward()
    return head_confidence(head_outputs)