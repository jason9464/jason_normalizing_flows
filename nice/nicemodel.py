"""
nicemodel.py
"""
import torch.nn as nn
import torch

class CouplingLayer(nn.Module):
    def __init__(self, immobile, input_dim, hidden_dim, layer_num):
        super(CouplingLayer, self).__init__()

        self.immobile = immobile

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, input_dim))  

        self.layer1 = nn.Sequential(*layers)
        for i in range(layer_num+1):
            nn.init.kaiming_uniform_(layers[2*i].weight, nonlinearity='relu')

    """
    1. Partitioning input tensor by it_even, it_odd
    2. set immobile tensor
    3. set fully connected layer on mobile tensor
    """
    def forward(self, x, inverse=False):
        immobile = self.immobile
        it_even = x[:,0::2].to("cuda")
        it_odd = x[:,1::2].to("cuda")

        if immobile == 'even':
            temp = self.layer1(it_even)
            if inverse is True:
                temp *= -1
            it_odd = it_odd + temp
        elif immobile == 'odd':
            temp = self.layer1(it_odd)
            if inverse is True:
                temp *= -1
            it_even = it_even + temp
        
        output_tensor = torch.zeros_like(x).to("cuda")

        output_tensor[:,0::2] = it_even
        output_tensor[:,1::2] = it_odd

        return output_tensor

class NICEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num):
        super(NICEModel, self).__init__()
        self.cl1 = CouplingLayer("odd", int(input_dim/2), hidden_dim, layer_num)
        self.cl2 = CouplingLayer("even", int(input_dim/2), hidden_dim, layer_num)
        self.cl3 = CouplingLayer("odd", int(input_dim/2), hidden_dim, layer_num)
        self.cl4 = CouplingLayer("even", int(input_dim/2), hidden_dim, layer_num)
        self.scaling_tensor = nn.Parameter(torch.ones(input_dim))
        
    def forward(self, x):
        x = self.cl1(x)
        x = self.cl2(x)
        x = self.cl3(x)
        x = self.cl4(x)
        x= x * torch.exp(self.scaling_tensor)

        return x

    def inverse(self, x):
        x = x*torch.exp(-self.scaling_tensor)
        x = self.cl4(x, inverse=True)
        x = self.cl3(x, inverse=True)
        x = self.cl2(x, inverse=True)
        x = self.cl1(x, inverse=True)

        return x
      