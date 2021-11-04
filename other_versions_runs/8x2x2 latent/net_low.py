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
    def __init__(self,  activation='leaky_r', last_activation='sigmoid'):
        super(AutoEncoder, self).__init__()
        self.activation = activations[activation]
        self.last_activation = activations[last_activation]
        self.enc1 = nn.Conv2d(3, 96, 11, stride=4, padding=4)
        self.enc1_bn = nn.BatchNorm2d(96)
        self.pool = nn.MaxPool2d(3, 2, padding=1,  return_indices=True)
        self.enc2 = nn.Conv2d(96, 256,  5, padding=2)
        self.enc2_bn = nn.BatchNorm2d(256)
        self.enc3 = nn.Conv2d(256, 384, 3)
        self.enc3_bn = nn.BatchNorm2d(384)
        self.enc4 = nn.Conv2d(384, 384, 3)
        self.enc4_bn = nn.BatchNorm2d(384)
        self.enc5 = nn.Conv2d(384, 256, 3, padding=1)
        self.enc5_bn = nn.BatchNorm2d(256)
        self.enc6 = nn.Conv2d(256, 16, 3)
        self.enc6_bn = nn.BatchNorm2d(16)
        self.enc7 = nn.Conv2d(16, 8, 3)
        self.enc7_bn = nn.BatchNorm2d(8)
        
        
        self.dec0 = nn.ConvTranspose2d(3, 3, 2)
        self.dec1 = nn.ConvTranspose2d(96, 3, 11, stride=4, padding=4)
        self.dec1_bn = nn.BatchNorm2d(3)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.dec2 = nn.ConvTranspose2d(256, 96,  5, padding=2)
        self.dec2_bn = nn.BatchNorm2d(96)
        self.dec3 = nn.ConvTranspose2d(384, 256, 3)
        self.dec3_bn = nn.BatchNorm2d(256)
        self.dec4 = nn.ConvTranspose2d(384, 384, 3)
        self.dec4_bn = nn.BatchNorm2d(384)
        self.dec5 = nn.ConvTranspose2d(256, 384, 3, padding=1)
        self.dec5_bn = nn.BatchNorm2d(384)
        self.dec6 = nn.ConvTranspose2d(16, 256, 3)
        self.dec6_bn = nn.BatchNorm2d(256)
        self.dec7 = nn.ConvTranspose2d(8, 16, 3)
        self.dec7_bn = nn.BatchNorm2d(16)
        
        
        
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc1_bn(x)
        x = self.activation(x)
        x, indices1 = self.pool(x)
        x = self.enc2(x)
        x = self.enc2_bn(x)
        x = self.activation(x)
        x, indices2 = self.pool(x)
        x = self.enc3(x)
        x = self.enc3_bn(x)
        x = self.activation(x)
        x = self.enc4(x)
        x = self.enc4_bn(x)
        x = self.activation(x)
        x = self.enc5(x)
        x = self.enc5_bn(x)
        x = self.activation(x)
        x, indices5 = self.pool(x)
        x = self.enc6(x)
        x = self.enc6_bn(x)
        x = self.activation(x)
        
        x = self.enc7(x)
        x = self.enc7_bn(x)
        x = self.activation(x)
        
        x = self.dec7(x)
        x = self.dec7_bn(x)
        x = self.activation(x)
        x = self.dec6(x)
        x = self.dec6_bn(x)
        x = self.activation(x)
        x = self.unpool(x, indices5)
        x = self.dec5(x)
        x = self.dec5_bn(x)
        x = self.activation(x)
        x = self.dec4(x)
        x = self.dec4_bn(x)
        x = self.activation(x)
        x = self.dec3(x)
        x = self.dec3_bn(x)
        x = self.activation(x)
        x = self.unpool(x, indices2)
        x = self.dec2(x)
        x = self.dec2_bn(x)
        x = self.activation(x)
        x = self.unpool(x, indices1)
        x = self.dec1(x)
        x = self.dec1_bn(x)
        x = self.activation(x)
        return self.last_activation(self.dec0(x))
    
  