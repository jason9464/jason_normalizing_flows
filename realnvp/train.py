import torch
import torch.optim as optim
import torchvision
import argparse
import make_datasets
import realnvp
import loss_function
import realnvp_utils as rut
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def set_variables(args):
    d_name = args.dataset
    batch_size = args.batch_size

    if d_name == 'cifar10':
        input_dim = (batch_size, 3, 32, 32)
        d_channel = 3
        feature_num = 32
        resblock_num = 2
    else:
        raise NotImplementedError

    return batch_size, input_dim, d_channel, feature_num, resblock_num

def sampling(args):
    trainloader, validloader, testloader = make_datasets.load_data(args)
    batch_size, img_dim, d_channel, feature_num, resblock_num = set_variables(args)

    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    iter_num = args.recall_iter

    realnvp_model = realnvp.RealNVP(img_dim, d_channel, feature_num, resblock_num).to(device)
    checkpoint = torch.load("./bin/param/realnvp_cifar10_model_" + \
        str(args.recall_iter) + ".pt")
    realnvp_model.load_state_dict(checkpoint['model_state_dict'])
    print("Load iter {}".format(iter_num))

    normal_sample = torch.normal(0,1,size=(1,d_channel,img_dim[0],img_dim[1])).to(device)
    backward_output = realnvp_model.backward(normal_sample) 
    forward_output, _ = realnvp_model(backward_output)
    print(torch.mean(normal_sample-forward_output))
    before_logit = rut.logit(backward_output,backward=True)
    rut.make_image(before_logit, './bin/sample/sample' + str(args.recall_iter) + '.png')
    zero_sample = torch.zeros(1,d_channel,img_dim[0],img_dim[1]).to(device)
    backward_output = realnvp_model.backward(normal_sample) 
    before_logit = rut.logit(backward_output,backward=True)
    rut.make_image(before_logit, './bin/sample_3.png')

    for i, datas in enumerate(tqdm(validloader), 0):
        inputs, _ = datas
        inputs = inputs.to(device)
        before_logit = rut.logit(inputs,backward=True)
        rut.make_image(before_logit, './bin/sample_1.png')
        outputs, scaling_log_det = realnvp_model(inputs)
        backward_output = realnvp_model.backward(outputs)
        before_logit1 = rut.logit(backward_output,backward=True)
        rut.make_image(before_logit1, './bin/sample_2.png')
        break



def valid(args):
    trainloader, validloader, testloader = make_datasets.load_data(args)
    batch_size, img_dim, d_channel, feature_num, resblock_num = set_variables(args)

    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    realnvp_model = realnvp.RealNVP(img_dim, d_channel, feature_num, resblock_num).to(device)
    
    iter_num = args.recall_iter

    checkpoint = torch.load("./bin/param/realnvp_cifar10_model_" + \
        str(args.recall_iter) + ".pt")
    realnvp_model.load_state_dict(checkpoint['model_state_dict'])
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

        print("Valid loss: {:.2f}".format(valid_loss))

def train(args):
    writer = SummaryWriter('runs')

    trainloader, validloader, testloader = make_datasets.load_data(args)
    batch_size, input_dim, d_channel, feature_num, resblock_num = set_variables(args)

    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    realnvp_model = realnvp.RealNVP(input_dim, d_channel, feature_num, resblock_num).to(device)
    

    optimizer = optim.Adam(realnvp_model.parameters(), weight_decay=5e-5)
    
    iter_num = args.recall_iter

    if iter_num != 0:
        checkpoint = torch.load("./bin/param/realnvp_cifar10_model_" + \
            str(args.recall_iter) + ".pt")
        realnvp_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Load iter {}".format(iter_num))

    normal_sample_64 = torch.normal(0,1,size=input_dim).to(device)

    total_loss = 0
    while 1:
        for _, datas in enumerate(trainloader, 0):
            inputs, _ = datas
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs, scaling_log_det = realnvp_model(inputs)

            batch_loss = -loss_function.log_likelihood(outputs, scaling_log_det, 'normal')
            mean_loss = torch.mean(batch_loss)

            total_loss += mean_loss
            #print("Iter {}, Loss: {:.2f}".format(iter_num, mean_loss))

            mean_loss.backward()
            optimizer.step()

            iter_num += 1
            
            if iter_num % 10 == 0:
                print("Iter {}, Loss: {:.2f}".format(iter_num, total_loss/10))
                total_loss = 0

                torch.save({
                    'model_state_dict' : realnvp_model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                }, "./bin/param/realnvp_" + args.dataset + "_model_" + str(iter_num) + ".pt")

                with torch.no_grad():
                    """normal_sample = torch.normal(0,1,size=(1,d_channel,img_dim[0],img_dim[1])).to(device)
                    backward_output = realnvp_model.backward(normal_sample) 
                    before_logit = rut.logit(backward_output,backward=True)
                    rut.make_image(before_logit, './bin/sample/sample' + str(iter_num) + '.png')"""

                    backward_output = realnvp_model.backward(normal_sample_64)
                    before_logit = rut.logit(backward_output,backward=True)
                    inputs_grid = torchvision.utils.make_grid(before_logit)
                    writer.add_image('image', inputs_grid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
        "RealNVP training program.")
    parser.add_argument("--dataset", type=str, default="cifar10", \
        help="Select dataset to train. expected value: [cifar10, imagenet, lsun, celeba]")
    parser.add_argument("--batch_size", type=int, default=64, \
        help="Number of data to learn at a time")
    parser.add_argument("--excute_mode", type=str, default="train", \
        help="Select execute mode. expected value: [train, valid, sampling]")
    parser.add_argument("--recall_iter", type=int, default=122410, \
        help="Select the number of recall iter. 0 to start new iter.")

    arguments = parser.parse_args()
    
    if arguments.excute_mode == 'train':
        train(arguments)
    elif arguments.excute_mode == 'valid':
        valid(arguments)
    elif arguments.excute_mode == 'sampling':
        sampling(arguments)
    else:
        raise NotImplementedError
