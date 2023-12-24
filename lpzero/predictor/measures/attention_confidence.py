import torch


# Attention Confidence metric (for both head and softmax)
def attention_confidence(outputs):
    metric_array = []
    for output in outputs:
        metric_array.append(torch.mean(torch.max(output, 1)[0]))

    summed = torch.tensor(0.0)
    for j in range(len(outputs)):
        summed += torch.nansum(metric_array[j])

    return summed.detach().item()


def attention_confidence_normalized(outputs):
    metric_array = []
    for output in outputs:
        metric_array.append(torch.mean(torch.max(output, 1)[0]))

    summed = torch.tensor(0.0)
    for j in range(len(metric_array)):
        summed += torch.nansum(metric_array[j])
    summed /= len(metric_array)

    return summed.detach().item()
