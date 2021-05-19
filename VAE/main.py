import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from utils.loss import KLDLoss
from network.AE import VariationAutoEncoder, FC_VAE


def select_device(device=''):
    ''' set up environment  '''
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    
    cuda = not cpu and torch.cuda.is_available()
    
    return torch.device('cuda:0' if cuda else 'cpu')  


def train(device_id='1'):
    # CUDA device
    device = select_device(device_id)
    # MNIST dataset
    transform = T.Compose([ T.Resize((64,64)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.5],std=[0.5])])

    train_data = MNIST(root='../MNIST', train=True, transform=transform, download=True)
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
    # model setting
    model = VariationAutoEncoder(in_channel=1, img_size=64, latent_dim=128)
    # model = FC_VAE(in_channel=1, img_size=28, latent_dim=2)
    model.to(device)
    # criterion
    mse_criterion = nn.MSELoss(reduction='sum').to(device)
    kld_criterion = KLDLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    ''' Training '''
    train_epochs = 50
    cur_epoch = 0
    cur_iter = 0
    trian_loss_list = []

    pth_save_dir = os.path.join('./checkpoints/VAE', 'minist')
    if not os.path.exists(pth_save_dir):
        os.mkdir(pth_save_dir)
    # ==== Train Loop ==== #
    model.train()

    while cur_epoch < train_epochs:
        # one epoch
        for _, (imgs, _) in enumerate(train_dataloader):
            optimizer.zero_grad()

            imgs = imgs.to(device, dtype=torch.float32)
            mu, log_var, rec = model(imgs)

            loss1 = mse_criterion(rec, imgs)
            loss2 = kld_criterion(mu, log_var)
            loss = loss1 + loss2

            loss.backward()
            
            trian_loss_list.append(loss.item())

            optimizer.step()

            print("epoch {} , iter {} : loss {}, mse_loss {}, kld_loss {}".format(cur_epoch, cur_iter,\
                                             loss.item(), loss1.item(), loss2.item()))
            
            cur_iter += 1
        
        # drawing loss curve
        _, ax1 = plt.subplots(figsize=(11, 8))
        ax1.plot(range(len(trian_loss_list)), trian_loss_list)
        ax1.set_title("Average train loss vs iterations")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("train Loss")
        plt.savefig(os.path.join(pth_save_dir, 'Train_loss_vs_iterations_minist.png'))
        plt.clf()
        plt.close()
        
        cur_epoch += 1

        if cur_epoch%10==0:
            torch.save(model.state_dict(), os.path.join(pth_save_dir, '%d_VAE.pt' % cur_epoch ))

        torch.save(model.state_dict(), os.path.join(pth_save_dir, 'last_VAE.pt'))

def test(device_id='1'):
    # CUDA device
    device = select_device(device_id)
    # MNIST dataset
    transform = T.Compose([ T.Resize((64,64)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.5],std=[0.5])])

    test_data = MNIST(root='../MNIST', train=False, transform=transform, download=True)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)
    # model setting
    model = VariationAutoEncoder(in_channel=1, img_size=64, latent_dim=128)
    # model = FC_VAE(in_channel=1, img_size=28, latent_dim=2)
    model.load_state_dict(torch.load('./checkpoints/VAE/minist/last_VAE.pt'))
    model.to(device)

    # test and vis
    vis_num = 50
    save_dir = './visualization/VAE/minist'
    model.eval()

    idx = 0

    for _, (imgs, _) in enumerate(test_dataloader):
        imgs = imgs.to(device, dtype=torch.float32)
        _, _, recs = model(imgs)

        if idx > vis_num:
            break

        for i in range(imgs.shape[0]):
            fig, axs = plt.subplots(1, 2)
            img = denormalization(imgs.detach().cpu().numpy()[i])
            r_img = denormalization(recs.detach().cpu().numpy()[i])
            
            axs[0].imshow(img, cmap ='gray')
            axs[0].set_title('Input Image',fontsize=12)
            axs[1].imshow(r_img, cmap ='gray')
            axs[1].set_title('Reconstruction Image',fontsize=12)

            fig.tight_layout()
            plt.savefig(os.path.join(save_dir, '{}.png'.format(idx)) )
            plt.clf()
            plt.close()
            idx += 1


def sampling(device_id='1'):
    # CUDA device
    device = select_device(device_id)
    # MNIST dataset
    transform = T.Compose([ T.Resize((64,64)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.5],std=[0.5])])

    test_data = MNIST(root='../MNIST', train=False, transform=transform, download=True)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)
    # model setting
    model = VariationAutoEncoder(in_channel=1, img_size=64, latent_dim=128)
    # model = FC_VAE(in_channel=1, img_size=28, latent_dim=2)
    model.load_state_dict(torch.load('./checkpoints/VAE/minist/last_VAE.pt'))
    model.to(device)

    # test and vis
    vis_num = 64
    save_dir = './visualization/VAE/minist'
    model.eval()

    v_imgs = model.sample(vis_num, device=device)

    save_image(v_imgs, "sample.png")


def denormalization(x):
    ''' 解归一化 '''
    print(x.shape)

    mean = np.array([0.5])
    std = np.array([0.5])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    print(x.shape)

    return x


if __name__ == '__main__':
    
    train()
    test()
    sampling()