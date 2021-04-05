"""
loss_functions.py
"""
import torch
import torch.nn as nn

"""
Change formula to avoid overflow issue
"""
def logistic_distribution(it, st, bs):
    softplus = nn.Softplus()
    logistic_log_likelihood = -(softplus(it)+softplus(-it))

    return logistic_log_likelihood
    