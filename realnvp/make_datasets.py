import torch
import torchvision
import torch.utils.data as data
import realnvp_utils as rut

def load_data(args):
    dataset_name = args.dataset
    batch_size = args.batch_size

    if dataset_name == "cifar10":
        cifar10_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x : x + torch.rand_like(x)/256.),
            torchvision.transforms.Lambda(lambda x : rut.rescale_tensor(x, 0, 1)),
            torchvision.transforms.Lambda(lambda x : rut.logit(x, 0.05))
        ])

        cifar10_trainset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, \
            transform=cifar10_transform, download=True)
        testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, \
            transform=cifar10_transform, download=True)

        cifar10_trainset_len = len(cifar10_trainset)
        trainlen = int(cifar10_trainset_len/6*5)
        validlen = cifar10_trainset_len - trainlen
        trainset, validset = data.dataset.random_split(cifar10_trainset, [trainlen, validlen])
        
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        validloader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)
        testloader = data.DataLoader(testset, batch_size=batch_size)

    elif dataset_name == "imagenet":
        raise NotImplementedError
    elif dataset_name == "LSUN":
        raise NotImplementedError
    elif dataset_name == "celeba":
        raise NotImplementedError

    return trainloader, validloader, testloader
