from . import measure


def num_parameters(model):
    return sum(p.numel() for p in model.parameters())
