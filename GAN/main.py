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

lib.init_params(gen)
lib.init_params(disc)

optim_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
optim_d = torch.optim.Adam(disc.parameters(), lr=1e-3)

for epoch in range(epochs):
    gen.train()
    disc.train()

    for (x, _) in TrainDataLoader:
        x = x.view(-1, 28*28)
        num_data = x.shape[0]
        noise = lib.sample_noise(num_data, num_noise)

        if torch.cuda.is_available():
            x = x.cuda()
            noise = noise.cuda()

        x_g = gen(noise)
        ### Generator train
        optim_g.zero_grad()
        loss_g = -torch.mean(torch.log(disc(x_g)))
        loss_g.backward()
        optim_g.step()

        ### Discriminator train
        optim_d.zero_grad()
        loss_d = -torch.mean(torch.log(disc(x)) + torch.log(1 - disc(x_g.detach())))
        loss_d.backward()
        optim_d.step()

    print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, epochs, loss_d.item(), loss_g.item()))

    if epoch % 10 == 9:
        gen.eval()
        noise = lib.sample_noise(24, num_noise)
        if torch.cuda.is_available():
            noise = noise.cuda()

        gen_img = gen(noise)
        #lib.imshow_grid(gen_img)
        lib.imsave(gen_img, epoch)