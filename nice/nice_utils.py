import torch

def rescale_tensor(input_tensor, low, high):
    input_high = torch.max(input_tensor)
    input_low = torch.min(input_tensor)

    assert input_high > input_low
    assert high > low

    output_interval = high-low
    alpha = (input_high*low+input_low*high)/output_interval
    beta = (input_low + input_high) / output_interval

    input_tensor = (input_tensor-alpha)/beta

    return input_tensor