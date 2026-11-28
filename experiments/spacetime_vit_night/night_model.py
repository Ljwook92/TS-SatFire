"""The project's factorized ViT with one full-resolution output for each time slot."""
import torch
from torch import nn
from torch.nn import functional as F
from model import SpaceTimeBlock


class NightViT(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        # 13 features, a daytime validity flag and a night-pair validity flag.
        self.patch = nn.Conv2d(15, width, 4, stride=4)
        self.position = nn.Parameter(torch.empty(1,1,64,64,width))
        self.day = nn.Parameter(torch.empty(1,4,1,1,width))
        nn.init.trunc_normal_(self.position, std=.02)
        nn.init.trunc_normal_(self.day, std=.02)
        self.blocks = nn.ModuleList(SpaceTimeBlock(width, shift) for shift in (0,4))
        self.heads = nn.ModuleList(nn.Sequential(nn.Conv2d(width+15,16,3,padding=1),
                                  nn.GELU(),nn.Conv2d(16,1,1)) for _ in range(2))

    def forward(self, x, task):
        batch, times, channels, h, w = x.shape
        assert (times, channels, h, w) == (4,15,256,256)
        flat = x.flatten(0,1)
        valid = F.max_pool2d(flat[:, -2:].amax(1,keepdim=True), 4).reshape(batch,4,64,64)>0
        z = self.patch(flat).permute(0,2,3,1).reshape(batch,4,64,64,-1)
        z = (z+self.position+self.day)*valid.unsqueeze(-1)
        for block in self.blocks:
            z = block(z, valid)
        z = F.interpolate(z.flatten(0,1).permute(0,3,1,2), size=(h,w),
                          mode='bilinear', align_corners=False)
        return self.heads[task](torch.cat([z, flat],1)).reshape(batch,4,h,w)
