

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from VAEGAN import VAE, VAE_CONV
from net import Discriminator, ConvDiscrim
from copy import deepcopy
import matplotlib.pyplot as plt
import random
import time
import cv2

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class VAECycleGan(nn.Module):

    def __init__(self, args):
        super(VAECycleGan, self).__init__()
        self.x_dim = args["x_dim"]
        self.z_dim = args["z_dim"]
        self.lam0 = args["lam0"]
        self.lam1 = args["lam1"]
        self.lam2 = args["lam2"]
        self.lam3 = args["lam3"]
        self.lam4 = args["lam4"]

        self.vae1 = VAE_CONV(self.x_dim, h_dim1 = 1024, h_dim2 = 1024, z_dim = self.z_dim).to(device)
        self.vae2 = VAE_CONV(self.x_dim, h_dim1 = 1024, h_dim2 = 1024, z_dim = self.z_dim).to(device)
        self.share_vae_features()

        self.D1 = ConvDiscrim(self.x_dim).to(device)
        self.D2 = ConvDiscrim(self.x_dim).to(device)
        
        self.G1 = self.vae1.Decoder
        self.G2 = self.vae2.Decoder



        self.X_dataloader = torch.utils.data.DataLoader(
            datasets.ImageFolder("horse2zebra/trainA", transform=transforms.Compose([
                transforms.Resize(128, interpolation=2),
                transforms.ToTensor(),
            ])),
        batch_size=64, shuffle=True)
        self.Y_dataloader = torch.utils.data.DataLoader(
            datasets.ImageFolder("horse2zebra/trainA", transform=transforms.Compose([
                transforms.Resize(128, interpolation=2),
                transforms.ToTensor(),
            ])),
        batch_size=64, shuffle=True)

        

        self.G_optim = optim.Adam(list(self.G1.parameters()) + list(self.G2.parameters()), lr=0.001)
        self.D_optim = optim.Adam(list(self.D1.parameters()) + list(self.D2.parameters()), lr=0.001)
        self.VAE_optim = optim.Adam(list(self.vae1.parameters()) + list(self.vae2.parameters()), lr=0.001)
        #self.VAE_optim = optim.Adam(list(self.vae1.parameters()) + list(self.vae2.parameters()))
        #self.VAE_optim = optim.Adam(self.vae1.parameters(), lr=0.001)


    def split_data(self, data, p_test):

        N = len(data)
        inds = list(range(N))
        random.shuffle(inds)

        train_N = int((1-p_test) * N)
        test_N = int((p_test * N))

        train = data[:train_N]
        test = data[train_N:]

        return train, test

    def share_vae_features(self):
        self.vae1.fc52 = self.vae2.fc52
        self.vae1.up1 = self.vae2.up1
        self.vae1.conv4 = self.vae2.conv4
        self.vae1.fc51 = self.vae2.fc51
        self.vae1.conv5 = self.vae2.conv5
        """
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.fc51 = nn.Linear(4*4*128, z_dim)
        self.fc52 = nn.Linear(4*4*128, z_dim)

        self.up1 = nn.Linear(z_dim, 4*4*128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        """

    def VAELoss(self, x_in, y_in):
        """
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
        """
        G1_out, mu1, log_var1 = self.vae1(x_in)
        KLD = -self.lam1 * (torch.mean(1 + log_var1 - mu1.pow(2) - log_var1.exp()))
        BCE = self.lam2 * (F.binary_cross_entropy(G1_out, x_in, reduction='mean'))
        L1 = (BCE + KLD)
        #print(G1_out.shape, log_var1.shape, mu1.pow(2).shape)


        G2_out, mu2, log_var2 = self.vae2(y_in)
        KLD_2 = -self.lam1 * (torch.mean(1 + log_var2 - mu2.pow(2) - log_var2.exp()))
        BCE_2 = self.lam2 * (F.binary_cross_entropy(G2_out, y_in, reduction='mean'))

        L = L1 + (BCE_2 + KLD_2)
        L.backward()

        return L

    def Disc_loss(self, x_in, y_in):
        ### Compute first loss term
        targets_real = torch.ones((x_in.shape[0], 1)).float().to(device)
        targets_fake = torch.zeros((x_in.shape[0], 1)).float().to(device)

        L1_D = self.lam0 * torch.mean(F.mse_loss(self.D1(x_in), targets_real))
        mu2, logvar2 = self.vae2.encode(y_in)
        G1out = self.G1(self.vae2.sampling(mu2.detach(), logvar2.detach()))
        disc_pred_1 = self.D1(G1out)
        L1_D = L1_D + self.lam0 * torch.mean(F.mse_loss(disc_pred_1, targets_fake))

        ### Compute second loss term
        L2_D = self.lam0 * torch.mean(F.mse_loss(self.D2(y_in), targets_real))
        mu1, logvar1 = self.vae1.encode(x_in)
        G2out = self.G2(self.vae1.sampling(mu1.detach(), logvar1.detach()))
        disc_pred_2 = self.D2(G2out)
        L2_D = L2_D + self.lam0 * torch.mean(F.mse_loss(disc_pred_2, targets_fake))

        L_D = L1_D + L2_D
        L_D.backward()

        return L_D

    def Gen_loss(self, x_in, y_in):

        targets_real = torch.ones((x_in.shape[0], 1)).float().to(device)

        mu2, logvar2 = self.vae2.encode(y_in)
        G1out = self.G1(self.vae2.sampling(mu2.detach(), logvar2.detach()))
        disc_pred_1 = self.D1(G1out)
        L1_G = self.lam0 * torch.mean(F.mse_loss(disc_pred_1, targets_real))

        mu1, logvar1 = self.vae1.encode(x_in)
        G2out = self.G2(self.vae1.sampling(mu1.detach(), logvar1.detach()))
        disc_pred_2 = self.D2(G2out)
        L2_G = self.lam0 * torch.mean(F.mse_loss(disc_pred_2, targets_real))

        L_G = L1_G + L2_G
        L_G.backward()

        return L_G


    def cycleConsistencyLoss(self, x_in, y_in):

        #G1_out, mu1, log_var1 = self.vae1(x_in)
        mu1, log_var1 = self.vae1.encode(x_in)
        G2_reconstr = self.vae2.decode(self.vae1.sampling(mu1, log_var1))
        mu2, log_var2 = self.vae2.encode(G2_reconstr)
        G121_cycle = self.vae1.decode(self.vae2.sampling(mu2, log_var2))

        L1 = -self.lam3 * (torch.mean(1 + log_var1 - mu1.pow(2) - log_var1.exp()))
        L1 = L1 - self.lam3 * (torch.mean(1 + log_var2 - mu2.pow(2) - log_var2.exp()))
        L1 = L1 + self.lam4 * (F.binary_cross_entropy(G121_cycle, x_in))

        mu2, log_var2 = self.vae2.encode(y_in)
        G1_reconstr = self.vae1.decode(self.vae2.sampling(mu2, log_var2))
        mu1, log_var1 = self.vae1.encode(G1_reconstr)
        G212_cycle = self.vae2.decode(self.vae1.sampling(mu1, log_var1))

        L2 = -self.lam3 * (torch.mean(1 + log_var2 - mu2.pow(2) - log_var2.exp()))
        L2 = L2 - self.lam3 * (torch.mean(1 + log_var1 - mu1.pow(2) - log_var1.exp()))
        L2 = L2 + self.lam4 * (F.binary_cross_entropy(G212_cycle, y_in))

        L = L1 + L2
        L.backward()

        return L

    def train(self, num_epochs):

        self.vae1.train()
        self.vae2.train()
        self.G1.train()
        self.G2.train()
        self.D1.train()
        self.D2.train()


        losses = []

        for i in range(num_epochs):
            epoch_loss = 0.0
            total_vae = 0.0
            total_D = 0.0
            total_G = 0.0
            total_cc = 0.0

            if (i == 30):
                self.G_optim = optim.Adam(list(self.G1.parameters()) + list(self.G2.parameters()), lr=0.0003)
                self.D_optim = optim.Adam(list(self.D1.parameters()) + list(self.D2.parameters()), lr=0.0003)
                self.VAE_optim = optim.Adam(list(self.vae1.parameters()) + list(self.vae2.parameters()), lr=0.0003)

            train_steps = min(len(self.X_dataloader), len(self.Y_dataloader))

            self.X_dataloader = torch.utils.data.DataLoader(
                datasets.ImageFolder("horse2zebra/trainA", transform=transforms.Compose([
                    transforms.Resize(128, interpolation=2),
                    transforms.ToTensor(),
                ])),
                batch_size=32, shuffle=True)
            self.Y_dataloader = torch.utils.data.DataLoader(
                datasets.ImageFolder("horse2zebra/trainA", transform=transforms.Compose([
                    transforms.Resize(128, interpolation=2),
                    transforms.ToTensor(),
                ])),
                batch_size=32, shuffle=True)

            x_data = iter(self.X_dataloader)
            y_data = iter(self.Y_dataloader)

            # Iterate through each batch
            for j in range(train_steps-1):
            #for j in range(10):
                # Get batch of data
                [x,_] = next(x_data)
                [y,_] = next(y_data)

                x = x.to(device)
                y = y.to(device)    

                #x = x.to(device).flatten(1, -1)
                #y = y.to(device).flatten(1,-1)
                #print(torch.max(x), torch.min(x))
                # Zero out all optimizers
                #self.G_optim.zero_grad()
                #self.D_optim.zero_grad()
                self.VAE_optim.zero_grad()


                # Sum loss functions
                # Gradient backpropagation computed inside these loss functions

                D_loss = self.Disc_loss(x, y)
                self.D_optim.step()
                self.D_optim.zero_grad()
                self.G_optim.zero_grad()

                G_loss = self.Gen_loss(x, y)
                self.G_optim.step()
                self.D_optim.zero_grad()
                self.G_optim.zero_grad()


                #G_loss, D_loss = torch.Tensor([0]), torch.Tensor([0])

                self.VAE_optim.zero_grad()
                vaeloss = self.VAELoss(x, y)
                self.VAE_optim.step()

                self.VAE_optim.zero_grad()
                ccloss = self.cycleConsistencyLoss(x, y)
                self.VAE_optim.step()
                #ccloss = torch.Tensor([0])
                # Exclude discriminator loss from total
                loss = vaeloss + ccloss

                epoch_loss += loss.item()
                total_vae += vaeloss.item()
                total_D += D_loss.item()
                total_G += G_loss.item()
                total_cc += ccloss.item()

                #self.G_optim.step()
                #self.D_optim.step()

            [epoch_loss, total_vae, total_D, total_G, total_cc] = loss_arr = np.array([epoch_loss, total_vae, total_D, total_G, total_cc]) / train_steps


            print("Losses at epoch %d\t VAE: %f\tDISC: %f\tGEN: %f\tCC: %f\tTOTAL: %f" % (i+1, total_vae, total_D, total_G, total_cc, epoch_loss))

            losses.append(list(loss_arr))
            plt.plot(np.array(losses)[:,0])
            plt.show(block=False)
            plt.pause(0.001)

        plt.figure()
        losses = np.array(losses)
        #[v, _, _, c] = plt.plot(losses[:,1:])
        [vae_loss_curve, cycle_consistency_loss_curve] = plt.plot(losses[:, [1,-1]])
        plt.legend([vae_loss_curve, cycle_consistency_loss_curve], ["VAE Loss", "Cycle Consistency Loss"], loc=1)

        plt.show()

    def test(self):

        self.vae1.eval()
        self.vae2.eval()
        self.G1.eval()
        self.G2.eval()
        self.D1.eval()
        self.D2.eval()


        [x,_] = next(iter(self.X_dataloader))
        x = x.to(device)
        #x = x.flatten(1,-1).to(device)
        trans, mu, log_var = self.vae1(x)
        trans = trans.reshape((-1, 3, 128, 128)).permute(0,2,3,1).cpu()[0].data.numpy()
        stylized = self.vae2.decode(self.vae1.sampling(mu, log_var)).reshape((-1,3,128,128)).permute(0,2,3,1).cpu()[0].data.numpy()

        #plt.imshow(x.reshape((-1,1,28,28)).cpu()[0,0].data.numpy())
        #plt.show()
        #plt.imshow(trans)
        #plt.show()
        plt.imshow(x.reshape((-1,3,128,128)).permute(0,2,3,1).cpu()[0].data.numpy())
        plt.show()
        plt.imshow(trans)
        plt.show()
        plt.imshow(stylized)
        plt.show()





def main():
    args = {
        "x_dim": 128*128*3,
        "z_dim": 10,
        "lam0": 1.0,            # Gan loss
        "lam1": 0.001,          # VAE KL loss
        "lam2": 2.0,            # VAE match loss
        "lam3": 0.001,           # CYCLEGAN KL loss
        "lam4": 2.0            # CYCLEGAN match loss
    }
    net = VAECycleGan(args)
    net.train(num_epochs=10)
    net.test()

    torch.save(net.state_dict(), "model.pth")






if __name__ == "__main__":
    main()
