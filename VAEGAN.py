# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

class VAE_CONV(nn.Module):

    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE_CONV, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.fc51 = nn.Linear(4*4*128, z_dim)
        self.fc52 = nn.Linear(4*4*128, z_dim)

        self.up1 = nn.Linear(z_dim, 4*4*128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.Encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.BatchNorm2d(32),
            self.conv2,
            nn.ReLU(),
            nn.BatchNorm2d(64),
            self.conv3,
            nn.ReLU(),
            nn.BatchNorm2d(128),
            self.conv4,
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((4,4)),
        )

        self.Decoder = nn.Sequential(
            self.up1,
            Reshape((-1,128,4,4)),
            self.conv5,
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            self.conv6,
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            self.conv7,
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            self.conv8,
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.Encoder(x)
        return self.fc51(h.flatten(1,-1)), self.fc52(h.flatten(1,-1)) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decode(self, z):
        output = self.Decoder(z)
        return output
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        return self.decode(z), mu, log_var

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.dropout = nn.Dropout(0.5)
        
        self.Encoder = nn.Sequential(
            self.fc1,
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim1),
            self.fc2,
            nn.Dropout(0.5),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim2)
        )
        
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
        self.Decoder = nn.Sequential(
            self.fc4,
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(h_dim2),
            #nn.ReLU(),
            self.fc5,
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(h_dim1),
            #nn.ReLU(),
            self.fc6,
            #nn.ReLU()
            nn.Sigmoid(),
        )

        
    def encode(self, x):
        h = self.Encoder(x)
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decode(self, z):
        output = self.Decoder(z)
        return output
    
    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        return self.decode(z), mu, log_var


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD
    
def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    bs = 100
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    # build model
    vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
    if torch.cuda.is_available():
        vae.cuda()
        
    optimizer = optim.Adam(vae.parameters())

    for epoch in range(1, 51):
        train(epoch)
        test()
        
    with torch.no_grad():
        z = torch.randn(64, 2).cuda()
        sample = vae.decode(z).cuda()
        
        save_image(sample.view(64, 1, 28, 28), './samples/sample_' + '.png')
