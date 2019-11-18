

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
from VAEGAN import VAE
from net import Discriminator
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
        
        self.vae1 = VAE(self.x_dim, h_dim1 = 2048, h_dim2 = 1024, z_dim = self.z_dim).to(device)
        self.vae2 = VAE(self.x_dim, h_dim1 = 2048, h_dim2 = 1024, z_dim = self.z_dim).to(device)
        #self.share_vae_features()
        
        self.D1 = Discriminator(self.x_dim).to(device)
        self.D2 = Discriminator(self.x_dim).to(device)
        self.G1 = self.vae1.Decoder
        self.G2 = self.vae2.Decoder
        
        
        self.young_data_fname = "kowalcyzk_logNorm_young_variableSubset.csv"
        self.old_data_fname = "kowalcyzk_logNorm_old_variableSubset.csv"
        self.young_data = np.genfromtxt(self.young_data_fname, delimiter=",").transpose()[1:,1:]
        self.old_data = np.genfromtxt(self.old_data_fname, delimiter=",").transpose()[1:,1:]
        
        
        """
        self.young_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=100, shuffle=True)
        self.old_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=100, shuffle=True)
        """
        
        
        self.young_data = torch.from_numpy(self.young_data).to(device).float()
        self.old_data = torch.from_numpy(self.old_data).to(device).float()
        
        
        self.young_train, self.young_test = self.split_data(self.young_data, 0.1)
        self.old_train, self.old_test = self.split_data(self.old_data, 0.1)
        
        
        self.young_ds = utils.TensorDataset(self.young_train)
        self.young_test_ds = utils.TensorDataset(self.young_test)
        self.young_dataloader = utils.DataLoader(self.young_ds, batch_size=10, shuffle=True)
        self.young_test_loader = utils.DataLoader(self.young_test_ds, batch_size=10, shuffle=True)
        self.old_ds = utils.TensorDataset(self.old_train)
        self.old_test_ds = utils.TensorDataset(self.old_test)
        self.old_dataloader = utils.DataLoader(self.old_ds, batch_size=10, shuffle=True)
        self.old_test_loader = utils.DataLoader(self.old_test_ds, batch_size=10, shuffle=True)
        
        
        self.G_optim = optim.Adam(list(self.G1.parameters()) + list(self.G2.parameters()), lr=0.005)
        self.D_optim = optim.Adam(list(self.D1.parameters()) + list(self.D2.parameters()), lr=0.005)
        self.VAE_optim = optim.Adam(list(self.vae1.parameters()) + list(self.vae2.parameters()), lr=0.005)
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
        self.vae1.fc31 = self.vae2.fc31
        self.vae1.fc32 = self.vae2.fc32
        self.vae1.fc4 = self.vae2.fc4
        
    def VAELoss(self, x_in, y_in):
        """
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
        """
        G1_out, mu1, log_var1 = self.vae1(x_in)
        KLD = -self.lam1 * (torch.mean(1 + log_var1 - mu1.pow(2) - log_var1.exp()))
        BCE = self.lam2 * (F.mse_loss(G1_out, x_in.view(-1, self.x_dim), reduction='mean'))
        L1 = (BCE + KLD)
        #print(G1_out.shape, log_var1.shape, mu1.pow(2).shape)
        
        
        G2_out, mu2, log_var2 = self.vae2(y_in)
        KLD_2 = -self.lam1 * (torch.mean(1 + log_var2 - mu2.pow(2) - log_var2.exp()))
        BCE_2 = self.lam2 * (F.mse_loss(G2_out, y_in.view(-1,self.x_dim), reduction='mean'))
        
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
        L1 = L1 + self.lam4 * (F.mse_loss(G121_cycle, x_in))
        
        mu2, log_var2 = self.vae2.encode(y_in)
        G1_reconstr = self.vae1.decode(self.vae2.sampling(mu2, log_var2))
        mu1, log_var1 = self.vae1.encode(G1_reconstr)
        G212_cycle = self.vae2.decode(self.vae1.sampling(mu1, log_var1))
        
        L2 = -self.lam3 * (torch.mean(1 + log_var2 - mu2.pow(2) - log_var2.exp()))
        L2 = L2 - self.lam3 * (torch.mean(1 + log_var1 - mu1.pow(2) - log_var1.exp()))
        L2 = L2 + self.lam4 * (F.mse_loss(G212_cycle, y_in))
        
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
                self.G_optim = optim.Adam(list(self.G1.parameters()) + list(self.G2.parameters()), lr=0.001)
            self.D_optim = optim.Adam(list(self.D1.parameters()) + list(self.D2.parameters()), lr=0.001)
            self.VAE_optim = optim.Adam(list(self.vae1.parameters()) + list(self.vae2.parameters()), lr=0.001)
        
            train_steps = min(len(self.old_dataloader), len(self.young_dataloader))
            
            self.young_dataloader = utils.DataLoader(self.young_ds, batch_size=10, shuffle=True)
            self.old_dataloader = utils.DataLoader(self.old_ds, batch_size=10, shuffle=True)
            old_data = iter(self.old_dataloader)
            young_data = iter(self.young_dataloader)
            
            """
            self.young_dataloader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                batch_size=100, shuffle=True)
            self.old_dataloader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
                batch_size=100, shuffle=True)
            
            young_data = iter(self.young_dataloader)
            old_data = iter(self.old_dataloader)
            """
            # Iterate through each batch
            for j in range(train_steps-1):
            #for j in range(10):   
                # Get batch of data
                [young_cells] = next(young_data)
                [old_cells] = next(old_data)
                
                young_cells = young_cells.to(device).flatten(1,-1)
                old_cells = old_cells.to(device).flatten(1,-1)
                #print(torch.max(young_cells), torch.min(young_cells))
                # Zero out all optimizers
                #self.G_optim.zero_grad()
                #self.D_optim.zero_grad()
                self.VAE_optim.zero_grad()
                
                """
                # Sum loss functions
                # Gradient backpropagation computed inside these loss functions
                
                D_loss = self.Disc_loss(young_cells, old_cells)
                self.D_optim.step()
                self.D_optim.zero_grad()
                self.G_optim.zero_grad()
                
                G_loss = self.Gen_loss(young_cells, old_cells)
                self.G_optim.step()
                self.D_optim.zero_grad()
                self.G_optim.zero_grad()
                """                
                
                G_loss, D_loss = torch.Tensor([0]), torch.Tensor([0])
                #vaeloss = torch.Tensor([0])
                # @TODO: Uncomment this block of code to enable VAE training
                  
                self.VAE_optim.zero_grad()
                vaeloss = self.VAELoss(young_cells, old_cells)
                self.VAE_optim.step()

                """                
                self.VAE_optim.zero_grad()
                ccloss = self.cycleConsistencyLoss(young_cells, old_cells)
                self.VAE_optim.step()
                """
                
                ccloss = torch.Tensor([0])
                # Exclude discriminator loss from total
                loss = vaeloss + G_loss + ccloss

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
            
            self.test(save=False)
            
        plt.figure()
        losses = np.array(losses)
        #[v, _, _, c] = plt.plot(losses[:,1:])
        [vae_loss] = plt.plot(losses[:, [1]])
        plt.legend([vae_loss], ["VAE Reconstruction"], loc=1)
        
        plt.show()
        
    def test(self, save=True):
        
        self.vae1.eval()
        self.vae2.eval()
        self.G1.eval()
        self.G2.eval()
        self.D1.eval()
        self.D2.eval()
        
        """
        young_mu, young_logvar = self.vae1.encode(self.young_test)
        young_Z = self.vae1.sampling(young_mu, young_logvar)
        young_output = self.vae1.decode(young_Z)
        young_corr = self.pearson_correlation(self.young_test, young_output)
        
        old_mu, old_logvar = self.vae2.encode(self.old_test)
        old_Z = self.vae2.sampling(old_mu, old_logvar)
        old_output = self.vae2.decode(old_Z)
        old_corr = self.pearson_correlation(self.old_test, old_output)
        
        
        print("old corr: ", old_corr, " young corr: ", young_corr)
        """
        
        
        
        young_mu, young_logvar = self.vae1.encode(self.young_data)
        young_Z = self.vae1.sampling(young_mu, young_logvar)
        young_output = self.vae1.decode(young_Z)
        young_corr = self.pearson_correlation(self.young_data, young_output)
        
        old_mu, old_logvar = self.vae2.encode(self.old_data)
        old_Z = self.vae2.sampling(old_mu, old_logvar)
        old_output = self.vae2.decode(old_Z)
        old_corr = self.pearson_correlation(self.old_data, old_output)
        
        print("old corr: ", old_corr, " young corr: ", young_corr)
        
        if (save):
            
            np.savetxt("old_mu.csv", old_mu.cpu().data.numpy())
            np.savetxt("old_logvar.csv", old_logvar.cpu().data.numpy())
            np.savetxt("old_Z.csv", old_Z.cpu().data.numpy())
            np.savetxt("old_correlation.csv", np.array([old_corr.cpu().data.numpy()]))
            np.savetxt("old_recreated_from_vae.csv", old_output.cpu().data.numpy())
            
            np.savetxt("young_mu.csv", young_mu.cpu().data.numpy())
            np.savetxt("young_logvar.csv", young_logvar.cpu().data.numpy())
            np.savetxt("young_Z.csv", young_Z.cpu().data.numpy())
            np.savetxt("young_correlation.csv", np.array([young_corr.cpu().data.numpy()]))
            np.savetxt("young_recreated_from_vae.csv", young_output.cpu().data.numpy())
        
        
        """
        print(list(self.young_test[0].cpu().data.numpy()), list(young_output[0].cpu().data.numpy()))
        print(list(self.old_test[0].cpu().data.numpy()), list(old_output[0].cpu().data.numpy()))
        """
            
        
        """
        [x,_] = next(iter(self.young_dataloader))
        x = x.flatten(1,-1).to(device)
        trans, mu, log_var = self.vae1(x)
        trans = trans.reshape((-1, 1, 28, 28)).cpu()[0,0].data.numpy()
        stylized = self.vae2.decode(self.vae1.sampling(mu, log_var)).reshape((-1,1,28,28)).cpu()[0,0].data.numpy()
        
        #plt.imshow(x.reshape((-1,1,28,28)).cpu()[0,0].data.numpy())
        #plt.show()
        #plt.imshow(trans)
        #plt.show()
        cv2.imshow("original", x.reshape((-1,1,28,28)).cpu()[0,0].data.numpy())
        cv2.waitKey(0)
        cv2.imshow("reconstructed", trans)
        cv2.waitKey(0)
        cv2.imshow("stylized", stylized)
        cv2.waitKey(0)
        """
            
        
        """
        young_zero_p = (torch.sum((self.young_data <= 0.0).float()) / self.young_data.numel())
        inferred_young_zero_p = (torch.sum(self.vae2(self.old_data)[0] <= 0).float() / self.old_data.numel())
        
        
        old_zero_p = (torch.sum((self.old_data <= 0.0).float()) / self.old_data.numel()).cpu().data.numpy()
        inferred_old_zero_p = (torch.sum(self.vae1(self.young_data)[0] <= 0).float() / self.young_data.numel()).cpu().data.numpy()
        
        print("Ground truth proportion of 0's in young data: %f. Predicted: %f." % (young_zero_p, inferred_young_zero_p))
        print("Ground truth proportion of 0's in old data: %f. Predicted: %f." % (old_zero_p, inferred_old_zero_p))
        
        mu1, log_var1 = self.vae2.encode(self.old_data)
        z1 = self.vae2.sampling(mu1, log_var1)
        inferred_young = self.vae1.decode(z1)
        mu2, log_var2 = self.vae1.encode(self.young_data)
        z2 = self.vae1.sampling(mu2, log_var2)
        inferred_old = self.vae2.decode(z2)
        
        np.savetxt("YoungToOld.csv", torch.clamp(inferred_old, 0.0).cpu().data.numpy())
        np.savetxt("OldToYoung.csv", torch.clamp(inferred_young, 0.0).cpu().data.numpy())
        np.savetxt("Old_latent.txt", z1.cpu().data.numpy())
        np.savetxt("Young_latent.txt", z2.cpu().data.numpy())
        
        
        mu1, log_var1 = self.vae1.encode(self.young_data[0])
        G2_reconstr = self.vae2.decode(self.vae1.sampling(mu1, log_var1))
        mu2, log_var2 = self.vae2.encode(G2_reconstr)
        G121_cycle = self.vae1.decode(self.vae2.sampling(mu2, log_var2))
        print(self.young_data[0], G2_reconstr, G121_cycle)
        """
        
    def pearson_correlation(self, x, y):
        normx = x - torch.mean(x)
        normy = y - torch.mean(y)
        
        return torch.mean( torch.sum(normx * normy, dim=1) / (torch.sqrt(torch.sum(normx ** 2, dim=1)) * torch.sqrt(torch.sum(normy ** 2, dim=1))) )
            

def main():
    args = {
        "x_dim": 2554,
        "z_dim": 10,
        "lam0": 5.0,            # Gan loss
        "lam1": 0.001,          # VAE KL loss
        "lam2": 1.0,            # VAE match loss
        "lam3": 0.001,           # CYCLEGAN KL loss
        "lam4": 5.0            # CYCLEGAN match loss
    }
    net = VAECycleGan(args)
    net.train(num_epochs=60)
    net.test()
    
    torch.save(net.state_dict(), "model.pth")
    





if __name__ == "__main__":
    main()





