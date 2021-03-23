"""
Todo

1. scaling matrix 구현하기
2. 같은 형태의 matrix라도 따로 layer를 구성해야 하는가?

"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class NICEModel(nn.Module):
    def __init__(self):
        super(NICEModel, self).__init__()
        self.fc1 = nn.Linear(392, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        self.fc6 = nn.Linear(1000, 1000)
        self.fc7 = nn.Linear(1000, 392)
        self.scaling_tensor = nn.Parameter(torch.zeros(28*28))
        
    def forward(self, x):
        x = self.coupling_layer(x, "even")
        x = self.coupling_layer(x, "odd")
        x = self.coupling_layer(x, "even")
        x = self.coupling_layer(x, "odd")
        x = torch.matmul(x, torch.diag(torch.exp(self.scaling_tensor)))
        return x

    def inverse(self, x):
        raise NotImplementedError

    """
    1. Partitioning input tensor by it_even, it_odd
    2. set immobile tensor
    3. set fully connected layer on mobile tensor
    """
    def coupling_layer(self, input_tensor, immobile):
        it_even = input_tensor[0::2]
        it_odd = input_tensor[1::2]

        if immobile == 'even':
            it_odd = F.relu(self.fc1(it_odd))
            it_odd = F.relu(self.fc2(it_odd))
            it_odd = F.relu(self.fc3(it_odd))
            it_odd = F.relu(self.fc4(it_odd))
            it_odd = F.relu(self.fc5(it_odd))
            it_odd = F.relu(self.fc6(it_odd))
            it_odd = F.relu(self.fc7(it_odd))
        elif immobile == 'odd':
            it_even = F.relu(self.fc1(it_even))
            it_even = F.relu(self.fc2(it_even))
            it_even = F.relu(self.fc3(it_even))
            it_even = F.relu(self.fc4(it_even))
            it_even = F.relu(self.fc5(it_even))
            it_even = F.relu(self.fc6(it_even))
            it_even = F.relu(self.fc7(it_even))

        return input_tensor