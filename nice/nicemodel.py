"""
Todo

"""
import torch.nn as nn
import torch

class CouplingLayer(nn.Module):
    def __init__(self, immobile):
        super(CouplingLayer, self).__init__()

        self.immobile = immobile

        layers = [nn.Linear(392, 1000), nn.ReLU()]
        for _ in range(5):
            layers.append(nn.Linear(1000, 1000))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(1000, 392))
        layers.append(nn.ReLU())     

        self.layer1 = nn.Sequential(*layers)

    """
    1. Partitioning input tensor by it_even, it_odd
    2. set immobile tensor
    3. set fully connected layer on mobile tensor
    """
    def forward(self, x):
        immobile = self.immobile
        it_even = x[:,0::2].to("cuda")
        it_odd = x[:,1::2].to("cuda")

        if immobile == 'even':
            temp = self.layer1(it_even)
            it_odd = it_odd + temp
        elif immobile == 'odd':
            temp = self.layer1(it_odd)
            it_even = it_even + temp
        
        output_tensor = torch.zeros_like(x).to("cuda")

        output_tensor[:,0::2] = it_even
        output_tensor[:,1::2] = it_odd

        return output_tensor

class NICEModel(nn.Module):
    def __init__(self):
        super(NICEModel, self).__init__()
        self.cl1 = CouplingLayer("odd")
        self.cl2 = CouplingLayer("even")
        self.cl3 = CouplingLayer("odd")
        self.cl4 = CouplingLayer("even")
        self.scaling_tensor = nn.Parameter(torch.ones(28*28))
        
    def forward(self, x):
        x = self.cl1(x)
        x = self.cl2(x)
        x = self.cl3(x)
        x = self.cl4(x)
        x= x * torch.exp(self.scaling_tensor)

        return x

    def inverse(self, x):
        raise NotImplementedError

    

      