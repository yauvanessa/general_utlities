import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 200
from torch.utils.data import DataLoader, Dataset


class Decoder(nn.Module):
    def __init__(self, latent_dims, device='cpu'):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 128)
        self.linear_2 = nn.Linear(128, 256)
        self.linear_3 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 1200)
        self.device = device

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear_2(z))
        z = F.relu(self.linear_3(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 1200))


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, device='cpu'):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(1200, 512)
        self.linear_1 = nn.Linear(512, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.linear_3 = nn.Linear(128, 64)
        self.linear_4 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, latent_dims)
        self.linear3 = nn.Linear(32, latent_dims)
        self.device = device

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc # hack to get sampling on the GPU
        self.N.scale = self.N.scale
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = F.relu(self.linear_4(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape).to(self.device)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, device='cpu'):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, device)
        self.decoder = Decoder(latent_dims, device)
        self.device = device

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class DiasTimeSeriesDataset(Dataset):
    """
       Defines the custom DIAS pytorch dataset
    """

    def __init__(self, time_series, labels: any = None, normalise: bool = True):
        self.X = time_series
        self.y = labels

        if normalise:

            self.normalise_data()

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        data = self.X[index, :]

        if self.y is not None:

            return (data, self.y[index])

        else:

            return (data)

    def normalise_data(self):

        offset = 0.5

        for idx in range(self.X.shape[0]):
            self.X[idx, :] -= self.X[idx, :].mean()
            # data /= data.std()

            max_value = np.max(np.abs(self.X[idx, :][0, 10:50]))

            # normalise the data
            self.X[idx, :] = (self.X[idx, :] / max_value) * 0.2 + offset
