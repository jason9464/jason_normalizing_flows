"""
train.py
"""
from nicemodel import NICEModel
import make_datasets
import loss_functions
import torch.optim as optim
import torch
import torch.utils.data as data
from tqdm import tqdm

PATH = './bin/nice_model.pt'
batch_size = 100

torch.autograd.set_detect_anomaly(True)
max_norm = 0.5 * batch_size

cuda = torch.device('cuda')
nm = NICEModel().cuda()
optimizer = optim.Adam(nm.parameters(), lr=0.001, betas=(0.9, 0.01), eps=0.0001, weight_decay=0)

mnist_trainset_loader = data.DataLoader(dataset=make_datasets.mnist_trainset, shuffle=True, batch_size=batch_size)

save = True
if save == True:
    checkpoint = torch.load(PATH)
    nm.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Saved epoch = {}".format(epoch))
else:
    epoch = 0

batch_loss = torch.zeros(1).to("cuda")
stacked_loss = 0
while epoch < 1500:
    print("Epoch {} start".format(epoch))
    for i, data in enumerate(tqdm(mnist_trainset_loader), 0):
        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.view(batch_size,-1)
        outputs = nm(inputs)

        loss = loss_functions.logistic_distribution(outputs, nm.scaling_tensor, batch_size)
        stacked_loss += loss.item()

        """if i%5 == 0:
            print("Loss: {}".format(loss/100))"""

        loss.backward()
        torch.nn.utils.clip_grad_norm_(nm.parameters(), max_norm)
        optimizer.step()

    print("Mean loss: {:.2f}".format(stacked_loss/60000))
    stacked_loss = 0

    epoch += 1
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : nm.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : loss,
    }, PATH)


        

print("Finish training")
