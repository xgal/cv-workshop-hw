import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

activations = nn.ModuleDict([
                ['leaky_r', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['sigmoid', nn.Sigmoid()],
                ['tanh', nn.Tanh()]
    ])

class AutoEncoder(nn.Module):
    def __init__(self, activation='leaky_r', last_activation='sigmoid'):
        super().__init__()
        self.activation = activations[activation]
        self.last_activation = activations[last_activation]
        self.enc1 = nn.Conv2d(3, 64, 5, 1, 2)
#         self.enc1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.enc1_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.enc2_bn = nn.BatchNorm2d(32)
        self.enc3 = nn.Conv2d(32, 16, 3, 1, 1)
        self.enc3_bn = nn.BatchNorm2d(16)
        self.enc4 = nn.Conv2d(16, 8, 3, 1, 1)
        self.enc4_bn = nn.BatchNorm2d(8)
        self.enc5 = nn.Conv2d(8, 4, 3, 1, 1)
        self.enc5_bn = nn.BatchNorm2d(4)
        # decoder 
        self.dec5 = nn.ConvTranspose2d(4, 8, 3, 1, 1)
        self.dec5_bn = nn.BatchNorm2d(8)
        self.dec4 = nn.ConvTranspose2d(8, 16, 3, 1, 1)
        self.dec4_bn = nn.BatchNorm2d(16)
        self.dec3 = nn.ConvTranspose2d(16, 32, 3, 1, 1)
        self.dec3_bn = nn.BatchNorm2d(32)
        self.dec2 = nn.ConvTranspose2d(32, 64, 3, 1, 1)
        self.dec2_bn = nn.BatchNorm2d(64)
        self.dec1 = nn.ConvTranspose2d(64, 3, 5, 1, 2)
#         self.dec1 = nn.ConvTranspose2d(64, 3, 3, 1, 1)
        self.dec1_bn = nn.BatchNorm2d(3)
        self.unpool = nn.MaxUnpool2d(2,2)
    def forward(self, x):
        x = self.activation(self.enc1_bn(self.enc1(x)))
        x, indices1 = self.pool(x)
        x = self.activation(self.enc2_bn(self.enc2(x)))
        x, indices2 = self.pool(x)
        x = self.activation(self.enc3_bn(self.enc3(x)))
        x, indices3 = self.pool(x)
        x = self.activation(self.enc4_bn(self.enc4(x)))
        x, indices4 = self.pool(x)
        x = self.activation(self.enc5_bn(self.enc5(x)))
        x, indices5 = self.pool(x)

        x = self.unpool(x, indices5)
        x = self.activation(self.dec5_bn(self.dec5(x)))
        x = self.unpool(x,indices4)
        x = self.activation(self.dec4_bn(self.dec4(x)))
        x = self.unpool(x, indices3)
        x = self.activation(self.dec3_bn(self.dec3(x)))
        x = self.unpool(x, indices2)
        x = self.activation(self.dec2_bn(self.dec2(x)))
        x = self.unpool(x, indices1)
        x = self.last_activation(self.dec1_bn(self.dec1(x)))
        return x