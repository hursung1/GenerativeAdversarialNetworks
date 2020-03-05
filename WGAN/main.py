import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import pyfiles.lib as lib
import pyfiles.models as models


num_noise = batch_size = 64
epochs = 200

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize([0.5], [0.5])])

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

gen = models.Generator_FC(num_noise)
disc = models.Discriminator_FC(28*28)

if torch.cuda.is_available():
    gen = gen.cuda()
    disc = disc.cuda()

optim_g = torch.optim.RMSprop(gen.parameters(), lr=5e-5)
optim_d = torch.optim.RMSprop(disc.parameters(), lr=5e-5)

for epoch in range(epochs):
    gen.train()
    disc.train()

    for i, (x, _) in enumerate(TrainDataLoader):
        x = x.view(-1, 28*28)
        num_data = x.shape[0]
        noise = lib.sample_noise(num_data, num_noise)

        if torch.cuda.is_available():
            x = x.cuda()
            noise = noise.cuda()

        x_g = gen(noise)
        
        ### Discriminator train
        optim_d.zero_grad()
        p_real = disc(x)
        p_fake = disc(x_g.detach())

        loss_d = torch.mean(p_fake) - torch.mean(p_real)
        loss_d.backward()
        optim_d.step()

        for params in disc.parameters():
            params.data.clamp_(-0.01, 0.01)

        if i % 5 == 4:
            ### Generator train
            optim_g.zero_grad()
            loss_g = -torch.mean(disc(x_g))
            loss_g.backward()
            optim_g.step()

    print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, epochs, loss_d.item(), loss_g.item()))

    if epoch % 10 == 9:
        gen.eval()
        noise = lib.sample_noise(24, num_noise)
        if torch.cuda.is_available():
            noise = noise.cuda()

        gen_img = gen(noise)
        #lib.imshow_grid(gen_img)
        lib.imsave(gen_img, epoch)