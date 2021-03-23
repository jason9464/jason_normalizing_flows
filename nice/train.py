"""
train.py
"""
from nicemodel import NICEModel
import make_datasets
import loss_functions
import torch.optim as optim

nm = NICEModel()
optimizer = optim.Adam(nm.parameters(), lr=0.001, betas=(0.9, 0.01), eps=0.0001)

for epoch in range(2):
    for i, data in enumerate(make_datasets.mnist_trainset, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = nm(inputs)
        loss = loss_functions.logistic_distribution(outputs, nm.scaling_tensor)
        print(loss)
        loss.backward()
        optimizer.step()
print("Finish training")
