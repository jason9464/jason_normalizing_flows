"""
Todo
1. Dequantize 하기
2. TFD dataset 찾기
"""
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import nice_utils as nut

# MNIST datasets has error now 
"""torchvision.datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]"""

mnist_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Lambda(lambda x : x.view(-1)),
      torchvision.transforms.Lambda(lambda x : x + torch.rand_like(x)/256.),
      torchvision.transforms.Lambda(lambda x : nut.rescale_tensor(x, 0, 1))
])
mnist_trainset = torchvision.datasets.MNIST(root="./datasets/mnist", train=True,\
     transform=mnist_transform, download=True)
mnist_trainset_loader = data.DataLoader(dataset=mnist_trainset, shuffle=True, batch_size=1)

mnist_testset = torchvision.datasets.MNIST(root="./datasets/mnist", train=False,\
     transform=mnist_transform, download=True)

#tfd_dataset

svhn_trainset = torchvision.datasets.SVHN(root="./datasets/svhn", split="train",\
      transform=transforms.ToTensor(), download=True)

cifar10_trainset = torchvision.datasets.CIFAR10(root="./datasets/cifar10", train=True,\
      download=True, transform=transforms.ToTensor())