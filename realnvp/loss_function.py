import torch
import torch.distributions as distributions

def log_likelihood(x, s_ld, prior):
    B, C, H, W = x.size()

    if prior == 'normal':
        prior_distribution = distributions.Normal(0,1)
    
    log_lh = torch.sum(prior_distribution.log_prob(x).view(B, -1), dim=1) + s_ld

    return log_lh

    


