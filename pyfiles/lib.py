import os
import torch
import torchvision
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

def permute_mnist(num_task, batch_size):
    """
    Returns PermutedMNISTDataLoaders

    Parameters
    ------------
    num_task: number of tasks\n
    batch_size: size of minibatch

    Returns
    ------------
    num_task numbers of TrainDataLoader, TestDataLoader

    """
    train_loader = {}
    test_loader = {}
    
    train_data_num = 0
    test_data_num = 0
    
    for i in range(num_task):
        shuffle_seed = np.arange(28*28)
        np.random.shuffle(shuffle_seed)
        
        train_PMNIST_DataLoader = PermutedMNISTDataLoader(train=True, shuffle_seed=shuffle_seed)
        test_PMNIST_DataLoader = PermutedMNISTDataLoader(train=False, shuffle_seed=shuffle_seed)
        
        train_data_num += train_PMNIST_DataLoader.getNumData()
        test_data_num += test_PMNIST_DataLoader.getNumData()
        
        train_loader[i] = torch.utils.data.DataLoader(
                train_PMNIST_DataLoader,
                batch_size=batch_size)
        
        test_loader[i] = torch.utils.data.DataLoader(
                test_PMNIST_DataLoader,
                batch_size=batch_size)
    
    return train_loader, test_loader, int(train_data_num/num_task), int(test_data_num/num_task)


def imsave(img, epoch, path='imgs'):
    assert type(img) is torch.Tensor
    if not os.path.isdir(path):
        os.mkdir(path)
        
    fig = torchvision.utils.make_grid(img.cpu().detach()).numpy()[0]
    scipy.misc.imsave(path+'/%03d.png'%(epoch+1), fig)
    
    
def imshow(img):
    #img = (img+1)/2    
    img = img.squeeze()
    np_img = img.numpy()
    print(np_img.shape)
    plt.imshow(np_img, cmap='gray')
    plt.show()

    
def imshow_grid(img):
    img = torchvision.utils.make_grid(img.cpu().detach())
    #img = (img+1)/2
    npimg = img.numpy()
    #npimg = np.transpose(img.numpy(), (1, 2, 0))
    print(npimg.shape)
    plt.imshow(npimg[0], cmap='gray')
    plt.show()
    
    
def sample_noise(batch_size, num_noise, option='uniform'):
    """
    Returns 
    """
    if torch.cuda.is_available():
        if option == 'uniform':
            return torch.randn(batch_size, num_noise).cuda()
        elif option == 'normal':
            return torch.rand(batch_size, num_noise).cuda()
    else:
        if option == 'uniform':
            return torch.randn(batch_size, num_noise)
        elif option == 'normal':
            return torch.rand(batch_size, num_noise)
    

def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            torch.nn.init.xavier_normal_(p)
        else:
            torch.nn.init.uniform_(p, 0.1, 0.2)

    
def gan_evaluate(**kwargs):
    batch_size = kwargs['batch_size']
    num_noise = kwargs['num_noise']
    cur_task = kwargs['cur_task']
    gen = kwargs['gen']
    disc = kwargs['disc']
    TestDataLoaders = kwargs['TestDataLoaders']
    
    p_real = p_fake = 0.0
    test_dataloaders = TestDataLoaders[:cur_task+1]
    
    gen.eval()
    disc.eval()
        
    for testloader in test_dataloaders:
        for image, _ in testloader:
            if torch.cuda.is_available():
                image = image.cuda()

            with torch.autograd.no_grad():
                p_real += (torch.sum(disc(image.unsqueeze(1))).item())/10000.
                p_fake += (torch.sum(disc(gen(sample_noise(batch_size, num_noise)))).item())/10000.

    return p_real, p_fake


def solver_evaluate(cur_task, gen, solver, ratio, TestDataLoaders):
    gen.eval()
    solver.eval()

    solver_loss = 0.0
    celoss = torch.nn.CrossEntropyLoss()
    test_dataloader = TestDataLoaders[:cur_task+1]

    for i, data in enumerate(test_dataloader):
        for image, label in data:
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()

            with torch.autograd.no_grad():
                output = solver(image)
                if i == cur_task:
                    solver_loss += celoss(output, label) * ratio
                
                else:
                    solver_loss += celoss(output, label) * (1 - ratio)

    return solver_loss