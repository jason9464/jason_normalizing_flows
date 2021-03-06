## DENSITY ESTIMATION USING Real NVP  
Pytorch implemantation of Laurent Dinh's paper [*Density estimation using Real NVP*](https://arxiv.org/abs/1605.08803)  


### Sampling
<img src = "./img/cifar10_50000.png" width="30%">  
Sampling data of CIFAR-10 dataset after 50000 iter<br><br>

<img src = "./img/mnist_sample.png" width="30%">  
Sampling data of MNIST dataset after 4000 iter    
  

### To use  
    train.py [-h] [--dataset DATASET] [--batch_size BATCH_SIZE] [--excute_mode EXCUTE_MODE] [--recall_iter RECALL_ITER]  

    RealNVP training program.  

    optional arguments:  
      -h, --help            show this help message and exit  
      --dataset DATASET     Select dataset to train. expected value: [cifar10, imagenet, lsun, celeba, mnist]  
      --batch_size BATCH_SIZE  
                            Number of data to learn at a time  
      --excute_mode EXCUTE_MODE  
                            Select execute mode. expected value: [train, valid, sampling]  
      --recall_iter RECALL_ITER  
                            Select the number of recall iter. 0 to start new iter.  


#### 1. Load datasets  
    CIFAR-10, Imagenet, LSUN, CelebA, MNIST  
#### 2. Data preprocessing  
##### 2-1 Reduce boundary effects  
    Use logit to reduce boundary effects
##### 2-2 horizontal filps  
    For CIFAR-10, CelebA, LSUN  
##### 2-3 Etc  
    Imagenet: 32*32, 64*64 downsampling  
    LSUN: Use bedroom, tower, church categories  
        Downsample image so that the smallest side is 96 pixels and take random crops of 64*64  
    CelebA: Take approximately central crop of 148*148 then resize it to 64*64  
#### 3. Model configure  
##### 3-1. Multiscale architecture  
    3 checkerboard masking -> squeezing -> 3 channel-wise masking -> 4 checkerboard masking  
    repeat until the input of the last coupling layer is 4*4*c  
##### 3-2. Coupling layer  
    x1 = x1  
    x2 = x2*exp(s(x1)) + t(x1)  

    s: NN with hyperbolic tangent function  
    t: NN with affine output  

    For 32*32 image, use 4 residual blocks with 32 feature maps  
    For 64*64 image, use 2 residual blocks with 32 feature maps  
    For CIFAR-10, use 8 residual blocks, 64 featire map, downsampling only once  
##### 3-3 Etc  
    batch_size = 64  
#### 4. Optimization  
##### 4-1 Optimizer  
    Use ADAM with default parameters  
    Use L2 normalization with 5e-5 weight decay  
##### 4-2 Loss  
    Isotropic unit norm gaussian  
