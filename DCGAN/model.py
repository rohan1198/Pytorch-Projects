import torch
import torch.nn as nn



class Generator(nn.Module):
    def __init__(self, input_dim = 100, output_channels = 1):
        super(Generator, self).__init__()

        self.ct1 = nn.ConvTranspose2d(in_channels = input_dim,
                                      out_channels = 128,
                                      kernel_size = 4,
                                      stride = 2,
                                      padding = 0,
                                      bias = False)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(128)

        self.ct2 = nn.ConvTranspose2d(in_channels = 128,
                                      out_channels = 64,
                                      kernel_size = 3,
                                      stride = 2,
                                      padding = 1,
                                      bias = False)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.ct3 = nn.ConvTranspose2d(in_channels = 64,
                                      out_channels = 32,
                                      kernel_size = 4,
                                      stride = 2,
                                      padding = 1,
                                      bias = False)
        self.relu3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(32)

        self.ct4 = nn.ConvTranspose2d(in_channels = 32,
                                      out_channels = output_channels,
                                      kernel_size = 4,
                                      stride = 2,
                                      padding = 1,
                                      bias = False)
        self.tanh = nn.Tanh()
    

    def forward(self, x):
        x = self.ct1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)

        x = self.ct2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)

        x = self.ct3(x)
        x = self.relu3(x)
        x = self.batchnorm3(x)

        x = self.ct4(x)
        output = self.tanh(x)

        return output



class Discriminator(nn.Module):
    def __init__(self, depth = 1, alpha = 0.2):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = depth, 
                               out_channels = 32, 
                               kernel_size = 4, 
                               stride = 2, 
                               padding = 1)
        self.leakyrelu1 = nn.LeakyReLU(alpha, inplace = True)

        self.conv2 = nn.Conv2d(in_channels = 32,
                               out_channels = 64, 
                               kernel_size = 4,
                               stride = 2,
                               padding = 1)
        self.leakyrelu2 = nn.LeakyReLU(alpha, inplace = True)

        self.fc1 = nn.Linear(in_features = 3136, out_features = 512)
        self.leakyrelu3 = nn.LeakyReLU(alpha, inplace = True)

        self.fc2 = nn.Linear(in_features = 512, out_features = 1)
        self.sigmoid = nn.Sigmoid()
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu1(x)

        x = self.conv2(x)
        x = self.leakyrelu2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.leakyrelu3(x)

        x = self.fc2(x)
        output = self.sigmoid(x)

        return output