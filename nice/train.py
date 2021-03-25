"""
train.py
"""
from nicemodel import NICEModel
import make_datasets
import loss_functions
import torch.optim as optim
import torch
import math

PATH = 'nice_model.pt'

torch.autograd.set_detect_anomaly(True)
max_norm = 0.5

cuda = torch.device('cuda')
nm = NICEModel().cuda()
optimizer = optim.Adam(nm.parameters(), lr=0.001, betas=(0.9, 0.01), eps=0.0001, weight_decay=0)

save = True
if save == True:
    """nm = NICEModel().cuda()
    optimizer = optim.Adam(nm.parameters(), lr=0.001, betas=(0.9, 0.01), eps=0.0001, weight_decay=0)"""
    checkpoint = torch.load(PATH)
    nm.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Saved epoch = {}",format(epoch))
    # model.eval()
    # or
    # model.train()
else:
    epoch = 0

batch_loss = torch.zeros(1).to("cuda")
stacked_loss = 0
while epoch < 10:
    for i, data in enumerate(make_datasets.mnist_trainset_loader, 0):
        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.view(-1)
        outputs = nm(inputs)

        loss = loss_functions.logistic_distribution(outputs, nm.scaling_tensor)
        stacked_loss += loss.item()

        if i % 5 == 0:
            print("Loss: {:.2f}, Epoch: {}, Iter: {}".format(stacked_loss/5, epoch, i))
            stacked_loss = 0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(nm.parameters(), max_norm)
        optimizer.step()

    epoch += 1
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : nm.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : loss,
    }, PATH)


        

print("Finish training")
