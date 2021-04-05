import argparse
import torch.optim as optim
import torch
import torch.utils.data as data
import torchvision
import tqdm
import make_datasets
import loss_functions
import numpy as np
import matplotlib.pyplot as plt
import nice_utils as nut
from nicemodel import NICEModel


def make_image(input_tensor, image_shape, path, device):
    """for i in range(len(input_tensor)):
        low = input_tensor[i][0]
        for j in range(len(input_tensor[i])):
            if input_tensor[i][j] <= low:
                input_tensor[i][j] = low
            else:
                input_tensor[i][j] -= low
        """

    batch_size = input_tensor.size()[0]
    for i in range(batch_size):
        input_tensor[i] = nut.rescale_tensor(input_tensor[i],0,256-(1e-3))
    input_tensor = input_tensor.int()

    image_tensor = torch.tensor([]).to(device)
    for i in range(image_shape[1]):
        line_tensor = input_tensor[int(batch_size/image_shape[1]*i):int(batch_size/image_shape[1]*(i+1))]
        line_image = line_tensor.view(image_shape[0]*28,-1)
        image_tensor = torch.cat((image_tensor, line_image), dim=1)
    image_array = image_tensor.to("cpu").numpy()

    plt.imshow(image_array,cmap="gray")
    # plt.show()
    plt.savefig(path)
    

def encode_output(encode_num_list, outputs):
    for batch_output in outputs:
        for value in batch_output:
            rounded_value = round(value.item(),1)
            if rounded_value >= 10 or rounded_value < -10:
                continue
            encode_num_list[int((rounded_value+10)*100)] += 1
    return encode_num_list


def validate(args):
    path = "./bin/nice_" + args.dataset + "_model.pt"
    batch_size = args.batch_size

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "mnist":
        mnist_trainset = torchvision.datasets.MNIST(root="../datasets/mnist", train=True, \
            transform=make_datasets.mnist_transform, download=True)
        trainset_loader = data.DataLoader(dataset=mnist_trainset, \
            shuffle=True, batch_size=batch_size)
        mnist_testset = torchvision.datasets.MNIST(root="../datasets/mnist", train=False, \
            transform=make_datasets.mnist_transform, download=True)
        testset_loader = data.DataLoader(dataset=mnist_testset, batch_size=batch_size)
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
    checkpoint = torch.load(path)
    nm.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Saved epoch = {}".format(epoch))

    print("Start validation task")
    total_loss = 0
    encode_num_list = [0 for i in range(2000)]
    for i, datas in enumerate(tqdm.tqdm(testset_loader, 0)):
        inputs, _ = datas

        if args.dataset == "svhn" or args.dataset == "cifar10":
            inputs = nut.zca_whitening(inputs,0,1)

        outputs = nm(inputs)

        log_likelihood = loss_functions.logistic_distribution(outputs, nm.scaling_tensor, batch_size)
        each_loss = -(log_likelihood + nm.scaling_tensor)
        batch_loss = torch.sum(each_loss, dim=1)
        loss = torch.sum(batch_loss)
        total_loss = total_loss + loss
        # encode_num_list = encode_output(encode_num_list, outputs)

        if i==0:
            number_tensor = torch.tensor([]).to(device)
            for i in range(100):
                log_randlist = np.random.logistic(size=[1,28*28])
                sample_tensor = nm.inverse(torch.tensor(log_randlist).float().to(device))
                number_tensor = torch.cat((number_tensor, sample_tensor))
            make_image(number_tensor, [10,10], "./sample/" + args.dataset + "_" + str(epoch) + ".png", device)


    print("Validation loss: {:.2f}".format(total_loss/10000))
    """np_encode_num_list = np.array(encode_num_list, float)
    np_encode_num_list = np_encode_num_list / np.sum(np_encode_num_list)

    x_axis = [i/100 - 10 for i in range(2000)]
    print(np.sum(np_encode_num_list))
    plt.plot(x_axis, np_encode_num_list)
    plt.show()"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
        "NICE training program. expected value : [mnist, tfd, svhn, cifar10]")
    parser.add_argument("--dataset", type=str, default="mnist", \
        help="Select dataset to train")
    parser.add_argument("--batch_size", type=int, default=100, \
        help="Number of data to learn at a time")
    parser.add_argument("--lr", type=float, default=0.001, \
        help="Learning rate for optimization")
    parser.add_argument("--b1", type=float, default=0.9, \
        help="Beta1 for AdaM optimizer")
    parser.add_argument("--b2", type=float, default=0.999, \
        help="Beta2 for AdaM optimizer")
    parser.add_argument("--eps", type=float, default=0.0001, \
        help="Epsilon for AdaM optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, \
        help="Weight decay for AdaM optimizer")
    parser.add_argument("--epoch", type=int, default=1500, \
        help="Set training epoch")
    parser.add_argument("--new_training", type=int, default=0, \
        help="0 to Start new training\n1 to continue existing training")
    
    arguments = parser.parse_args()
    validate(arguments)