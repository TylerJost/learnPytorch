import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, zSize=10):
        super(Model, self).__init__()
        self.zSize = zSize

    def create(self, opts):
        self.scale_factor = 8 / (512 / opts.imsize)
        self.nLatentDims = opts.nLatentDims
        self.nChIn = opts.nChIn
        self.nChOut = opts.nChOut
        self.nOther = opts.nOther
        self.dropoutRate = opts.dropoutRate
        self.opts = opts

        self.create_autoencoder()
        self.create_adversary()
        self.assemble()

    def create_autoencoder(self):
        scale = self.scale_factor

        # Create encoder (generator)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.nChIn, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            nn.Conv2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            nn.Flatten(),
            nn.PReLU(),
            nn.Linear(1024 * scale * scale, self.nLatentDims),
            nn.BatchNorm1d(self.nLatentDims)
        )

        # Create decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.nLatentDims, 1024 * scale * scale),
            nn.Unflatten(1, (1024, scale, scale)),
            nn.PReLU(),
            nn.ConvTranspose2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        )

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def create_adversary(self):
        # Create adversary (discriminator)
        noise = 0.1
        self.adversary = nn.Sequential(
            nn.Linear(self.nLatentDims, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.adversary.apply(weights_init)

    def assemble(self):
        self.autoencoder = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x):
        return self.autoencoder(x), self.adversary(x)