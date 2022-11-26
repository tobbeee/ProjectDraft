import torch.nn as nn
import torch.nn.functional as F


class hor_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        u1 = nn.Upsample(scale_factor= 2,mode='linear') #Trying ot figure out the tensor 
        self.up1 = u1(in_ch)
        self.relu  = nn.ReLU()
        self.up2 = nn.Upsample(out_ch, out_ch, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, ch=(1,2,3,4,5)):
        super().__init__()
        self.enc = nn.ModuleList([hor_block(ch[i], ch[i+1]) for i in range(len(ch)-1)])
        self.pool       = nn.AvgPool3d(3, stride = (2,2,2))    #Random Numbers, not sure about the stride

    def forward(self, x):
        feat = []
        for enc in self.enc:
            x = enc(x)
            feat.append(x) 
            x = self.pool(x)
        return feat


class Decoder(nn.Module):
    def __init__(self, ch=(5,4,3,2,1)):
        super().__init__()
        self.enc = nn.ModuleList([hor_block(ch[i], ch[i+1]) for i in range(len(ch)-1)])
        self.pool       = nn.AvgPool3d(3, stride = (2,2,2))    #Random Numbers, not sure about the stride

    def forward(self, x):
        feat = []
        for enc in self.enc:
            x = enc(x)
            feat.append(x) 
            x = self.pool(x)
        return feat


class UNet(nn.Module):
    def __init__(self, enc=(1,2,3,4,5,6), dec=(6,5,4,3,2,1), num_class=10):
        super().__init__()
        self.encoder     = Encoder(enc)
        self.decoder     = Decoder(dec)
        self.head        = nn.AvgPool3d(dec[-1], num_class, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        return out