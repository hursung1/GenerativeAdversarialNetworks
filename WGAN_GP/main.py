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
ld = 10

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

optim_g = torch.optim.Adam(gen.parameters(), lr=1e-3, betas=(0, 0.9))
optim_d = torch.optim.Adam(disc.parameters(), lr=1e-3, betas=(0, 0.9))

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

        ## Regularization Term
        eps = torch.rand(1).item()
        x_hat = (x.detach().clone() * eps + x_g.detach().clone().view(x_g.shape[0], -1) * (1 - eps)).requires_grad_(True)

        loss_xhat = disc(x_hat)
        fake = torch.ones(loss_xhat.shape[0], 1).requires_grad_(False)
        if torch.cuda.is_available():
            fake = fake.cuda()
            
        gradients = torch.autograd.grad(
            outputs = loss_xhat,
            inputs = x_hat,
            grad_outputs=fake,
            create_graph = True,
            retain_graph = True,
            only_inputs = True
        )[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * ld
        '''
        loss_d_xhat.backward()
        grad_x_hat = x_hat.grad.data.detach()
        reg = ((grad_x_hat ** 2).sum().sqrt() - 1) ** 2 * ld

        torch.autograd.grad()
        '''
        p_real = disc(x)
        p_fake = disc(x_g.detach())

        loss_d = torch.mean(p_fake) - torch.mean(p_real) + gp
        loss_d.backward()
        optim_d.step()

        if i % 5 == 4:
            ### Generator train
            optim_g.zero_grad()
            p_fake = disc(x_g)
            loss_g = -torch.mean(p_fake)
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