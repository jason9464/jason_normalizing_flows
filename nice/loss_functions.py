import torch

def logistic_distribution(input_tensor, scaling_tensor):
    return torch.sum(-1 * torch.log(1 + torch.exp(input_tensor)) - torch.log(1 + torch.exp(-1 * input_tensor)))
