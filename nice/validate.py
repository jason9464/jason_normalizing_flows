import torch
import numpy as np
import matplotlib.pyplot as plt
import nice_utils as nut

def make_image(input_tensor, image_shape, path, device):
    batch_size = input_tensor.size()[0]
    for i in range(batch_size):
        over_range = input_tensor[i] > 1
        under_range = input_tensor[i] < 0
        input_tensor[i][over_range] = 1
        input_tensor[i][under_range] = 0

        input_tensor[i] = nut.rescale_tensor(input_tensor[i],0,256-(1e-3))
    input_tensor = input_tensor.int()

    image_tensor = torch.tensor([]).to(device)
    for i in range(image_shape[1]):
        line_tensor = input_tensor[int(batch_size/image_shape[1]*i):\
            int(batch_size/image_shape[1]*(i+1))]
        line_image = line_tensor.view(image_shape[0]*28,-1)
        image_tensor = torch.cat((image_tensor, line_image), dim=1)
    image_array = image_tensor.to("cpu").numpy()

    plt.imshow(image_array,cmap="gray")
    plt.savefig(path)

def encode_output(encode_num_list, outputs):
    for batch_output in outputs:
        rounded_value = round(batch_output[int(28*28/3)].item(),1)
        if rounded_value >= 10 or rounded_value < -10:
            continue
        encode_num_list[int((rounded_value+10)*100)] += 1
    return encode_num_list

def make_encode_num_fig(encode_num_list):
    np_encode_num_list = np.array(encode_num_list, float)
    np_encode_num_list = np_encode_num_list / np.sum(np_encode_num_list)

    x_axis = [i/100 - 10 for i in range(2000)]
    plt.plot(x_axis, np_encode_num_list)
    plt.savefig('./sample/encode_num.png')
