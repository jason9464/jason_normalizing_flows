from nicemodel import NICEModel
import make_datasets
import loss_functions
import torch
import torch.optim as optim
import time

nm = NICEModel()
optimizer = optim.Adam(nm.parameters(), lr=0.001, betas=(0.9, 0.01), eps=0.0001)

for epoch in range(2):
    for i, data in enumerate(make_datasets.mnist_trainset, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = nm(inputs)
        loss = loss_functions.logistic_distribution(outputs)
        print(loss)
        loss.backward()
        optimizer.step()

        """if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))"""

        
print("Finish training")
    
        
