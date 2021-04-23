import torch
import argparse
import make_datasets
import realnvp
import loss_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
        "RealNVP training program.")
    parser.add_argument("--dataset", type=str, default="cifar10", \
        help="Select dataset to train. expected value : [cifar10, imagenet, lsun, celeba]")
    parser.add_argument("--batch_size", type=int, default=64, \
        help="Number of data to learn at a time")

    arguments = parser.parse_args()
    tl, _, _ = make_datasets.load_data(arguments)
    dataiter = iter(tl)
    a, b = dataiter.next()

    c = realnvp.RealNVP(32,3,64,8)
    out, s_ld = c.forward(a)
    loss = torch.mean(-loss_function.log_likelihood(out, s_ld, 'normal'))

    print(loss)