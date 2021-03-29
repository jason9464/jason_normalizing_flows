"""
train.py
"""
import argparse
import torch.optim as optim
import torch
import torch.utils.data as data
import tqdm
import loss_functions
import make_datasets
from nicemodel import NICEModel


def train(args):
    path = './bin/nice_model.pt'
    batch_size = args.batch_size

    torch.autograd.set_detect_anomaly(True)
    max_norm = 0.5 * batch_size

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    nm = NICEModel().to(device)
    optimizer = optim.Adam(nm.parameters(), lr=args.lr, betas=(args.b1, args.b2), \
        eps=args.eps, weight_decay=args.weight_decay)

    if args.dataset == "mnist":
        trainset_loader = data.DataLoader(dataset=make_datasets.mnist_trainset, \
            shuffle=True, batch_size=batch_size)

    if args.new_training:
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
            inputs = inputs.view(batch_size,-1)
            outputs = nm(inputs)

            loss = loss_functions.logistic_distribution(outputs, nm.scaling_tensor, batch_size)
            stacked_loss += loss.item()

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
    parser = argparse.ArgumentParser(description="NICE training program")
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
    parser.add_argument("--new_training", type=int, required=True, \
        help="0 to Start new training\n1 to continue existing training")

    arguments = parser.parse_args()
    print(arguments)
    train(arguments)
