import torch
import matplotlib.pyplot as plt

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

def logit(x, alpha=0.05, backward=False):
    if backward:
        logit_input = torch.sigmoid(x)
        """ret = (logit_input-alpha)/(1-2*alpha)
        ret = rescale_tensor(ret, 0, 1)"""
        return torch.sigmoid(x)
    else:
        logit_input = alpha + (1-2*alpha)*x
        
        return torch.log(logit_input) - torch.log(1-logit_input)

def squeeze(x):
    x1 = x[:,:,0::2,0::2]
    x2 = x[:,:,0::2,1::2]
    x3 = x[:,:,1::2,0::2]
    x4 = x[:,:,1::2,1::2]

    x = torch.cat((x1,x2,x3,x4),dim=1)

    return x

def unsqueeze(x):
    B, C, H, W = x.size()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    C = C//4

    y = torch.zeros(B,C,2*H,2*W).to(device)
    y[:,:,0::2,0::2] = x[:,:C]
    y[:,:,0::2,1::2] = x[:,C:2*C]
    y[:,:,1::2,0::2] = x[:,2*C:3*C]
    y[:,:,1::2,1::2] = x[:,3*C:4*C]

    return y

def make_image(input_tensor, path):
    B, C, H, W = input_tensor.size()

    input_tensor = input_tensor.permute(0,2,3,1)
    input_tensor = (input_tensor * (256-(1e-3))).int()
    image_numpy = input_tensor.to('cpu').numpy()

    for i in range(B):
        plt.imshow(image_numpy[i])
    plt.savefig(path)