import torch
import math

"""
Change formula to avoid overflow issue
"""
def logistic_distribution(it, st):
    inf_tensor = ~torch.isfinite(torch.exp(it))
    loss_tensor = torch.zeros_like(it)
    loss_tensor[~inf_tensor] = -torch.log1p(torch.exp(it[~inf_tensor]))-torch.log1p(torch.exp(-it[~inf_tensor])) + st[~inf_tensor]
    loss_tensor[inf_tensor] = -it[inf_tensor] -torch.log1p(torch.exp(-it[inf_tensor])) + st[inf_tensor]

    return_loss = -torch.sum(loss_tensor)
    return return_loss
    