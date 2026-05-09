import torch


def is_sm10x():
    return torch.rtriton.get_device_capability() >= (10, 0)


def is_hopper():
    return torch.rtriton.get_device_capability() == (9, 0)
