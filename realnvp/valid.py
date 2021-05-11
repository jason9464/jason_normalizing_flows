import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
        "RealNVP training program.")
    parser.add_argument("--dataset", type=str, default="cifar10", \
        help="Select dataset to train. expected value: [cifar10, imagenet, lsun, celeba]")
    parser.add_argument("--batch_size", type=int, default=64, \
        help="Number of data to learn at a time")
    parser.add_argument("--excute_mode", type=str, default="valid", \
        help="Select execute mode. expected value: [train, valid, sampling]")
    parser.add_argument("--recall_iter", type=int, default=100000, \
        help="Select the number of recall iter. 0 to start new iter.")

    arguments = parser.parse_args()

    max_iter = arguments.recall_iter
    min_loss = float('inf')

    loss_list = [0, 0, 0]

    right_valid_loss = train.valid(arguments)
    loss_list[2] = arguments.recall_iter

    arguments.recall_iter = 0
    left_valid_loss = train.valid(arguments)
    loss_list[0] = arguments.recall_iter

    arguments.recall_iter = max_iter//2
    mid_valid_loss = train.valid(arguments)
    loss_list[1] = arguments.recall_iter
    

    while 1:
        min_loss = min(loss_list)
        min_iter = loss_list.index(min_loss)

        
