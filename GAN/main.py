import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import pyfiles.lib as lib
import pyfiles.models as models


batch_size = 64
num_noise = 100
epochs = 200
data_shape = (1, 28, 28)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

MNISTTrainDataset = torchvision.datasets.MNIST(root='../data', 
                                                train=True, 
                                                download=True, 
                                                transform=transform)

MNISTTestDataset = torchvision.datasets.MNIST(root='../data', 
                                               train=False, 
                                               download=True,
                                               transform=transform)

TrainDataLoader = torch.utils.data.DataLoader(MNISTTrainDataset, batch_size=batch_size, shuffle=True)
TestDataLoader = torch.utils.data.DataLoader(MNISTTestDataset, batch_size=batch_size, shuffle=False)

gen = models.Generator_Conv(input_node_size=num_noise, output_shape=data_shape)
disc = models.Discriminator_Conv(input_shape=data_shape)
bceloss = torch.nn.BCELoss()

if torch.cuda.is_available():
    gen = gen.cuda()
    disc = disc.cuda()
    bceloss = bceloss.cuda()

lib.init_params(gen)
lib.init_params(disc)

optim_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
optim_d = torch.optim.Adam(disc.parameters(), lr=1e-3)

for epoch in range(epochs):
    gen.train()
    disc.train()

    for (x, _) in TrainDataLoader:
        #x = x.view(-1, 28*28)
        num_data = x.shape[0]
        noise = lib.sample_noise(num_data, num_noise)
        zeros = torch.zeros(num_data, 1)
        ones = torch.ones(num_data, 1)

        if torch.cuda.is_available():
            x = x.cuda()
            noise = noise.cuda()
            zeros = zeros.cuda()
            ones = ones.cuda()

        x_g = gen(noise)

        ### Discriminator train
        optim_d.zero_grad()
        disc.zero_grad()

        p_real = disc(x)
        p_fake = disc(x_g.detach())
        loss_d = bceloss(p_real, ones) + bceloss(p_fake, zeros)
        #loss_d = -torch.mean(torch.log(disc(x)) + torch.log(1 - disc(x_g.detach())))
        
        loss_d.backward()
        optim_d.step()
        
        ### Generator train
        gen.zero_grad()
        optim_g.zero_grad()

        p_fake = disc(x_g.detach())
        loss_g = bceloss(p_fake, ones)
        #loss_g = -torch.mean(torch.log(disc(x_g)))
        
        loss_g.backward()
        optim_g.step()


    print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch+1, epochs, loss_d.item(), loss_g.item()))

    if epoch % 10 == 9:
        gen.eval()
        noise = lib.sample_noise(24, num_noise)
        if torch.cuda.is_available():
            noise = noise.cuda()

        gen_img = gen(noise)
        lib.imsave(gen_img, epoch)