import torch
import torch.optim as optim
import argparse
import make_datasets
import realnvp
import loss_function
from tqdm import tqdm

def set_variables(args):
    d_name = args.dataset
    batch_size = args.batch_size

    if d_name == 'cifar10':
        img_dim = (32, 32)
        d_channel = 3
        feature_num = 64
        resblock_num = 8
    else:
        raise NotImplementedError

    return batch_size, img_dim, d_channel, feature_num, resblock_num

def valid(args):
    trainloader, validloader, testloader = make_datasets.load_data(args)
    batch_size, img_dim, d_channel, feature_num, resblock_num = set_variables(args)

    torch.autograd.set_detect_anomaly(True)
    """if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")"""
    device = device = torch.device("cpu")

    realnvp_model = realnvp.RealNVP(img_dim, d_channel, feature_num, resblock_num).to(device)
    optimizer = optim.Adam(realnvp_model.parameters(), weight_decay=5e-5)
    
    iter_num = 0

    checkpoint = torch.load("./bin/param/realnvp_cifar10_model_" + \
        str(args.recall_epoch) + ".pt")
    realnvp_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter_num = checkpoint['iter_num']
    print("Load iter {}".format(iter_num))

    total_loss = 0
    with torch.no_grad():
        for i, datas in enumerate(tqdm(validloader), 0):
            inputs, _ = datas
            inputs = inputs.to(device)
            outputs, scaling_log_det = realnvp_model(inputs)

            batch_loss = -loss_function.log_likelihood(outputs, scaling_log_det, 'normal')
            mean_loss = torch.mean(batch_loss)
            total_loss += mean_loss

        valid_loss = total_loss / len(validloader)

        print("Valid loss: {}".format(valid_loss))

def train(args):
    trainloader, validloader, testloader = make_datasets.load_data(args)
    batch_size, img_dim, d_channel, feature_num, resblock_num = set_variables(args)

    torch.autograd.set_detect_anomaly(True)
    """if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")"""
    device = device = torch.device("cpu")

    realnvp_model = realnvp.RealNVP(img_dim, d_channel, feature_num, resblock_num).to(device)
    optimizer = optim.Adam(realnvp_model.parameters(), weight_decay=5e-5)
    
    iter_num = 0

    checkpoint = torch.load("./bin/param/realnvp_cifar10_model_" + \
        str(args.recall_epoch) + ".pt")
    realnvp_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter_num = checkpoint['iter_num']
    print("Load iter {}".format(iter_num))

    while 1:
        for i, datas in enumerate(trainloader, 0):
            inputs, _ = datas
            inputs = inputs.to(device)
            outputs, scaling_log_det = realnvp_model(inputs)

            batch_loss = -loss_function.log_likelihood(outputs, scaling_log_det, 'normal')
            mean_loss = torch.mean(batch_loss)

            print("Iter {}, Loss: {:.2f}".format(iter_num, mean_loss))

            optimizer.zero_grad()
            mean_loss.backward()
            optimizer.step()

            iter_num += 1
            
            if iter_num % 10 == 0:
                torch.save({
                    'iter_num' : iter_num,
                    'model_state_dict' : realnvp_model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                }, "./bin/param/realnvp_" + args.dataset + "_model_" + str(iter_num) + ".pt")





    loss = torch.mean(-loss_function.log_likelihood(out, s_ld, 'normal'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
        "RealNVP training program.")
    parser.add_argument("--dataset", type=str, default="cifar10", \
        help="Select dataset to train. expected value: [cifar10, imagenet, lsun, celeba]")
    parser.add_argument("--batch_size", type=int, default=64, \
        help="Number of data to learn at a time")
    parser.add_argument("--excute_mode", type=str, default="train", \
        help="Select execute mode. expected value: [train, valid, sampling]")
    parser.add_argument("--recall_epoch", type=int, default=3020, \
        help="Select the number of recall epoch. 0 to start new epoch.")

    arguments = parser.parse_args()
    
    if arguments.excute_mode == 'train':
        train(arguments)
    elif arguments.excute_mode == 'valid':
        valid(arguments)
    elif arguments.excute_mode == 'sampling':
        raise NotImplementedError
    else:
        raise NotImplementedError
