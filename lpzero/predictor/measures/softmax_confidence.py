import torch
from . import measure

from lpzero.model.flexibert.modeling_electra import (
    ElectraConfig,
    ElectraLayer,
    ElectraModel,
)


# Attention Confidence metric (for both head and softmax)
def attention_confidence(outputs):
    metric_array = []
    for output in outputs:
        metric_array.append(torch.mean(torch.max(output, 1)[0]))
    
    summed = torch.tensor(0.0).to("cuda")
    for j in range(len(outputs)):
        summed += torch.nansum(metric_array[j])
        
    return summed.detach().item()

def compute_softmax_confidence(model, inputs):
    outputs = model(**inputs).last_hidden_state

    softmax_outputs = []
    def softmax_hook(module, input, output):
        softmax_outputs.append(output)

    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            sublayer = layer.operation.operation
            if hasattr(sublayer, 'softmax'):
                sublayer.softmax.register_forward_hook(softmax_hook)
    
    return attention_confidence(softmax_outputs)
