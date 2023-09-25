# -*- coding: utf-8 -*-

from torch import nn

def Encoder():
    
    enc = nn.Sequential(
        
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2))
        )
    return enc


def Decoder():
    
    dec = nn.Sequential(
        
        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(2, 2), stride=2),
        nn.Sigmoid()
    )
    return dec


class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, input_images):
        encoder_out = self.encoder(input_images)
        decoder_out = self.decoder(encoder_out)
        
        return decoder_out