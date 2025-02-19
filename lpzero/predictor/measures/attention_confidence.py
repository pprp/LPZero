import torch
from lpzero.model.flexibert.modeling_electra import (
    ElectraConfig,
    ElectraLayer,
    ElectraModel,
)
from lpzero.model.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex
import transformers 

from . import measure

# Attention Confidence metric (for both head and softmax)
def attention_confidence(outputs):
    metric_array = []
    for output in outputs:
        metric_array.append(torch.mean(torch.max(output, 1)[0]))

    summed = torch.tensor(0.0).to('cuda')
    for j in range(len(outputs)):
        summed += torch.nansum(metric_array[j])

    return summed.detach().item()


def attention_confidence_normalized(outputs):
    metric_array = []
    for output in outputs:
        metric_array.append(torch.mean(torch.max(output, 1)[0]))

    summed = torch.tensor(0.0).to('cuda')
    for j in range(len(metric_array)):
        summed += torch.nansum(metric_array[j])
    summed /= len(metric_array)

    return summed.detach().item()

@measure('attention_confidence')
def compute_attention_confidence(model, inputs, targets=None, *args, **kwargs):
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

        # Run gradient with respect to ones
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

        # Run gradient with respect to ones
        loss, _, _, _ = model.forward(inputs, targets, mems=None)
        loss = loss.float().mean().type_as(loss)
        loss.backward()
    
    return attention_confidence(head_outputs)

@measure('attention_importance')
def compute_attention_importance(model, inputs, targets=None, *args, **kwargs):
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

        # Run gradient with respect to ones
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

        # Run gradient with respect to ones
        loss, _, _, _ = model.forward(inputs, targets, mems=None)
        loss = loss.float().mean().type_as(loss)
        loss.backward()
    
    return attention_confidence_normalized(head_outputs)