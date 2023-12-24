import torch


# Activation Distance metric
def activation_distance(outputs):
    metric_array = []
    for output in outputs:
        output = output[0].view(output.size(1), -1)
        x = (output > 0).float()
        K = x @ x.t()
        K2 = (1.0 - x) @ (1.0 - x.t())
        metric_array.append(K + K2)

    summed = torch.tensor(0.0).to('cuda')
    for j in range(len(outputs)):
        summed += torch.nansum(metric_array[j])

    return summed.detach().item()


def activation_distance_normalized(outputs):
    metric_array = []
    for output in outputs:
        output = output[0].view(output.size(1), -1)
        x = (output > 0).float()
        K = x @ x.t()
        K2 = (1.0 - x) @ (1.0 - x.t())
        metric_array.append(K + K2)

    summed = torch.tensor(0.0).to('cuda')
    for j in range(len(outputs)):
        summed += torch.nansum(metric_array[j])
    summed /= len(metric_array)

    return summed.detach().item()
