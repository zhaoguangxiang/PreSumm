import torch


def logical_not(tensor):
    if torch.__version__ == '1.1.0':
        return 1-tensor
    elif torch.__version__ == '1.2.0':
        return ~tensor
    else:
        return torch.logical_not(tensor)