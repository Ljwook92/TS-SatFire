"""Small factorized space/time Transformer and two frozen-feature Bernoulli heads."""
import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, channels=47, width=32, latent=8):
        super().__init__()
        self.patch=nn.Conv2d(channels,width,16,16)
        self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,64,dropout=0.,batch_first=True,norm_first=True),1,enable_nested_tensor=False)
        self.spatial=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,64,dropout=0.,batch_first=True,norm_first=True),2,enable_nested_tensor=False)
        self.space_pos=nn.Parameter(torch.randn(1,256,width)*.02)
        self.time_pos=nn.Parameter(torch.randn(1,6,width)*.02)
        # Break identical cloned-layer initialization explicitly.
        for mod in self.modules():
            if isinstance(mod,nn.Linear):nn.init.xavier_uniform_(mod.weight)
        self.skip=nn.Sequential(nn.Conv2d(channels,8,3,padding=1),nn.GELU())
        self.decode=nn.Sequential(nn.Conv2d(width+8,16,3,padding=1),nn.GELU(),nn.Conv2d(16,latent,1),nn.Tanh())
        self.readout=nn.Conv2d(latent,1,1)
    def forward(self,x):
        b,t,c,h,w=x.shape
        assert h==w==256 and 1<=t<=6
        z=self.patch(x.reshape(b*t,c,h,w)).flatten(2).transpose(1,2).reshape(b,t,256,-1)
        z=z.permute(0,2,1,3).reshape(b*256,t,-1)+self.time_pos[:,:t]
        z=self.temporal(z)[:,-1].reshape(b,256,-1)
        z=self.spatial(z+self.space_pos).transpose(1,2).reshape(b,-1,16,16)
        z=F.interpolate(z,size=(256,256),mode='bilinear',align_corners=False)
        latent=self.decode(torch.cat([z,self.skip(x[:,-1])],dim=1))
        return latent,self.readout(latent).squeeze(1)
