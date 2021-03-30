"""
train.py
"""
import argparse
import torch.optim as optim
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import tqdm
import loss_functions
import make_datasets
import nice_utils as nut
from nicemodel import NICEModel


def train(args):
    path = "./bin/nice_" + args.dataset + "_model.pt"
    batch_size = args.batch_size

    torch.autograd.set_detect_anomaly(True)
    max_norm = 0.5 * batch_size

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "mnist":
        # MNIST datasets has HTTP error now 
        """torchvision.datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', \
                  'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', \
                  'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', \
                  '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', \
                  'ec29112dd5afa0611ce80d1b7f02629c')
        ]"""
        mnist_trainset = torchvision.datasets.MNIST(root="../datasets/mnist", train=True, \
            transform=make_datasets.mnist_transform, download=True)
        trainset_loader = data.DataLoader(dataset=mnist_trainset, \
            shuffle=True, batch_size=batch_size)
        input_dim = 28*28
        hidden_dim = 1000
        layer_num = 5
    elif args.dataset == "tfd":
        raise NotImplementedError
    elif args.dataset == "svhn":
        svhn_trainset = torchvision.datasets.SVHN(root="../datasets/svhn", split="train",\
            transform=make_datasets.svhn_transform, download=True)
        trainset_loader = data.DataLoader(dataset=svhn_trainset, \
            shuffle=True, batch_size=batch_size)
        input_dim = 3072
        hidden_dim = 2000
        layer_num = 4
    elif args.dataset == "cifar10":
        cifar10_trainset = torchvision.datasets.CIFAR10(root="../datasets/cifar10", train=True,\
            transform=make_datasets.cifar10_transform, download=True)
        trainset_loader = data.DataLoader(dataset=cifar10_trainset, \
            shuffle=True, batch_size=batch_size)
        input_dim = 3072
        hidden_dim = 2000
        layer_num = 4

    nm = NICEModel(input_dim, hidden_dim, layer_num).to(device)
    optimizer = optim.Adam(nm.parameters(), lr=args.lr, betas=(args.b1, args.b2), \
        eps=args.eps, weight_decay=args.weight_decay)

    if not args.new_training:
        checkpoint = torch.load(path)
        nm.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Saved epoch = {}".format(epoch))
    else:
        epoch = 0

    while epoch < args.epoch:
        print("Epoch {} start".format(epoch))
        for _, datas in enumerate(tqdm.tqdm(trainset_loader), 0):
            optimizer.zero_grad()
            inputs, _ = datas

            if args.dataset == "svhn" or args.dataset == "cifar10":
                inputs = nut.zca_whitening(inputs,0,1)
            outputs = nm(inputs)

            loss = loss_functions.logistic_distribution(outputs, nm.scaling_tensor, batch_size)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(nm.parameters(), max_norm)
            optimizer.step()

        epoch += 1
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : nm.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : loss,
        }, path)

    print("Finish training")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
        "NICE training program. expected value : [mnist, tfd, svhn, cifar10]")
    parser.add_argument("--dataset", type=str, default="mnist", \
        help="Select dataset to train")
    parser.add_argument("--batch_size", type=int, default=1, \
        help="Number of data to learn at a time")
    parser.add_argument("--lr", type=float, default=0.001, \
        help="Learning rate for optimization")
    parser.add_argument("--b1", type=float, default=0.9, \
        help="Beta1 for AdaM optimizer")
    parser.add_argument("--b2", type=float, default=0.01, \
        help="Beta2 for AdaM optimizer")
    parser.add_argument("--eps", type=float, default=0.0001, \
        help="Epsilon for AdaM optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, \
        help="Weight decay for AdaM optimizer")
    parser.add_argument("--epoch", type=int, default=1, \
        help="Set training epoch")
    parser.add_argument("--new_training", type=int, default=1, \
        help="0 to Start new training\n1 to continue existing training")

    arguments = parser.parse_args()
    train(arguments)
