import torch


def logical_not(tensor):
    return 1-tensor if torch.__version__ == '1.2.0' else torch.logical_not(tensor)