import torch.nn as nn
import math

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.15):
        super(Autoencoder, self).__init__()
        dim = list(input_dim)
        for i in range(3):
            dim[i] = dim[i] - 1
            dim[i] = math.ceil(dim[i] / 2)
            dim[i] = dim[i] - 1
            dim[i] = math.ceil(dim[i] / 2)
        self.dropout = nn.Dropout(dropout)
        self.c1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.m1 = nn.MaxPool3d(2, stride=2, return_indices=True)
        self.c2 = nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1)
        self.m2 = nn.MaxPool3d(2, stride=2, return_indices=True)
        self.f = nn.Flatten()
        self.l = nn.Linear(dim[0] * dim[1] * dim[2] * 8, hidden_dim)
        unfSize = (8, dim[0], dim[1], dim[2])
        self.dl = nn.Linear(hidden_dim, dim[0] * dim[1] * dim[2] * 8)
        self.uf = nn.Unflatten(1, unfSize)
        self.mu1 = nn.MaxUnpool3d(2, stride=2)
        self.dc1 = nn.ConvTranspose3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.mu2 = nn.MaxUnpool3d(2, stride=2)
        self.dc2 = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.dropout(x)
        x = self.c1(x)
        x = self.relu(x)
        s1 = x.shape
        x, i1 = self.m1(x)
        x = self.c2(x)
        x = self.relu(x)
        s2 = x.shape
        x, i2 = self.m2(x)
        x = self.f(x)
        x = self.l(x)
        x = self.relu(x)
        return x, (i1, i2, s1, s2)

    def forward(self, x):
        x, (i1, i2, s1, s2) = self.encode(x)
        x = self.dl(x)
        x = self.relu(x)
        x = self.uf(x)
        x = self.mu1(x, i2, output_size=s2)
        x = self.dc1(x)
        x = self.relu(x)
        x = self.mu2(x, i1, output_size=s1)
        x = self.dc2(x)
        return x