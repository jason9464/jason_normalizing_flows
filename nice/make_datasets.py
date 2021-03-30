"""
make_datasets.py

Todo
1. TFD dataset 찾기
"""
import torchvision
import torch
import nice_utils as nut

mnist_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Lambda(lambda x : x.view(-1)),
      torchvision.transforms.Lambda(lambda x : x + torch.rand_like(x)/256.),
      torchvision.transforms.Lambda(lambda x : nut.rescale_tensor(x, 0, 1))
])

svhn_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Lambda(lambda x : x.view(-1)),
      torchvision.transforms.Lambda(lambda x : x + torch.rand_like(x)/256.),
      torchvision.transforms.Lambda(lambda x : nut.rescale_tensor(x, 0, 1)),
])

cifar10_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Lambda(lambda x : x.view(-1)),
      torchvision.transforms.Lambda(lambda x : x + torch.rand_like(x)/128.),
      torchvision.transforms.Lambda(lambda x : nut.rescale_tensor(x, -1, 1)),
])  