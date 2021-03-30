"""
nice_utils.py
"""
import torch
import numpy as np

def rescale_tensor(input_tensor, low, high):
    input_high = torch.max(input_tensor)
    input_low = torch.min(input_tensor)

    assert input_high > input_low
    assert high > low

    output_interval = high-low
    alpha = (input_high*low+input_low*high)/output_interval
    beta = (input_high - input_low) / output_interval

    input_tensor = (input_tensor-alpha)/beta

    return input_tensor

def zca_whitening(input_tensor, low, high):
    batch_size = input_tensor.size()[0]
    unbiased_tensor = input_tensor - torch.mean(input_tensor, dim=1)
    unbiased_tensor_np = unbiased_tensor.numpy()
    cov_mat = (unbiased_tensor_np.T @ unbiased_tensor_np) / batch_size
    U, s, V = np.linalg.svd(cov_mat)

    # To avoid data distortion, add epsilon 1e-5
    # 1e-5 is empirical value
    eps = 1e-5
    L = np.diag(1./(np.sqrt(s+eps)))
    zca_mat = U @ L @ U.T

    output_tensor_T = torch.tensor(zca_mat) @ input_tensor.T
    output_tensor = output_tensor_T.T
    
    for i in range(batch_size):
        output_tensor[i] = rescale_tensor(output_tensor[i], low, high)

    return output_tensor
