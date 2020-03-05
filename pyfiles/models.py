import torch
import torchvision
import numpy as np


class Generator_FC(torch.nn.Module):
    """
    Fully-Connected Generator
    """
    def __init__(self, input_node_size, hidden_node_size=256, output_node_size=28*28):
        super(Generator_FC, self).__init__()
        self.network = torch.nn.Sequential(
            
            torch.nn.Linear(input_node_size, hidden_node_size),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Linear(hidden_node_size, hidden_node_size),
            torch.nn.BatchNorm1d(hidden_node_size, 0.8),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Linear(hidden_node_size, hidden_node_size),
            torch.nn.BatchNorm1d(hidden_node_size, 0.8),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Linear(hidden_node_size, hidden_node_size),
            torch.nn.BatchNorm1d(hidden_node_size, 0.8),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Linear(hidden_node_size, output_node_size),
            torch.nn.Tanh()
            
        )
        
    def forward(self, x):
        return self.network(x).view((x.shape[0], 1, 28, 28))


class Generator(torch.nn.Module):
    """
    Generator Class for GAN
    """
    def __init__(self, num_noise):
        super(Generator, self).__init__()
        conv2d_1 = torch.nn.ConvTranspose2d(in_channels=num_noise,
                                   out_channels=28*8, 
                                   kernel_size=7, 
                                   stride=1,
                                   padding=0,
                                   bias=False)
        conv2d_2 = torch.nn.ConvTranspose2d(in_channels=28*8, 
                                   out_channels=28*4, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_3 = torch.nn.ConvTranspose2d(in_channels=28*4, 
                                   out_channels=1, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)

        self.network = torch.nn.Sequential(
            conv2d_1,
            torch.nn.BatchNorm2d(num_features = 28*8),
            torch.nn.ReLU(inplace=True),
            conv2d_2,
            torch.nn.BatchNorm2d(num_features = 28*4),
            torch.nn.ReLU(inplace=True),
            conv2d_3,
            torch.nn.Tanh()
        )

        if torch.cuda.is_available():
            self.network = self.network.cuda()

        self.num_noise = num_noise
        
    def forward(self, x):
        return self.network(x.view(-1, self.num_noise, 1, 1))


class Discriminator_FC(torch.nn.Module):
    """
    Fully-Connected Discriminator
    """
    def __init__(self, input_node_size=28*28, hidden_node_size=256, output_node_size=1):
        super(Discriminator_FC, self).__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_node_size, hidden_node_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
            torch.nn.Linear(hidden_node_size, hidden_node_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
            torch.nn.Linear(hidden_node_size, output_node_size),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        _x = x.view(x.shape[0], -1)
        return self.network(_x).view(-1, 1)


class Discriminator(torch.nn.Module):
    """
    Discriminator Class for GAN
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        conv2d_1 = torch.nn.Conv2d(in_channels=1, 
                                   out_channels=28*4, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_2 = torch.nn.Conv2d(in_channels=28*4, 
                                   out_channels=28*8, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_3 = torch.nn.Conv2d(in_channels=28*8, 
                                   out_channels=1, 
                                   kernel_size=7, 
                                   stride=1,
                                   padding=0,
                                   bias=False)

        self.network = torch.nn.Sequential(
            conv2d_1,
            torch.nn.BatchNorm2d(num_features=28*4),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_2,
            torch.nn.BatchNorm2d(num_features=28*8),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_3,
            torch.nn.Sigmoid()
        )

        if torch.cuda.is_available():
            self.network = self.network.cuda()

    def forward(self, x):
        return self.network(x).view(-1, 1)

    
class Solver(torch.nn.Module):
    """
    Solver Class for Deep Generative Replay
    """
    def __init__(self, T_n):
        super(Solver, self).__init__()
        fc1 = torch.nn.Linear(28*28, 100)
        fc2 = torch.nn.Linear(100, 100)
        fc3 = torch.nn.Linear(100, T_n * 2)
        self.network = torch.nn.Sequential(
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.ReLU(),
            fc3
        )

        if torch.cuda.is_available():
            self.network = self.network.cuda()

    def forward(self, x):
        return self.network(x.view(x.shape[0], -1))
    