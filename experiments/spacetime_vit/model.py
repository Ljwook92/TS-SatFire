"""Factorized spatial-window and temporal attention with full-resolution segmentation."""
import torch
from torch import nn
from torch.nn import functional as F

class SpaceTimeBlock(nn.Module):
    def __init__(self,width,offset):
        super().__init__()
        self.offset=offset
        self.space=nn.MultiheadAttention(width,4,batch_first=True)
        self.time=nn.MultiheadAttention(width,4,batch_first=True)
        self.norm=nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))

    @staticmethod
    def attend(attn,norm,x,valid):
        # Keep one dummy key for completely missing windows; discard missing outputs.
        keys=valid.clone()
        keys[:,0]|=~valid.any(1)
        z=norm(x)
        out=attn(z,z,z,key_padding_mask=~keys,need_weights=False)[0]
        return (x+out)*valid.unsqueeze(-1)

    def forward(self,x,mask):
        b,t,h,w,c=x.shape
        # Offset windows by padding instead of wrapping opposite map edges.
        p=self.offset
        if p:
            x=F.pad(x,(0,0,p,p,p,p))
            mask=F.pad(mask,(p,p,p,p))
        H,W=x.shape[2:4]
        z=x.reshape(b,t,H//8,8,W//8,8,c).permute(0,1,2,4,3,5,6).reshape(-1,64,c)
        m=mask.reshape(b,t,H//8,8,W//8,8).permute(0,1,2,4,3,5).reshape(-1,64)
        z=self.attend(self.space,self.norm[0],z,m)
        x=z.reshape(b,t,H//8,W//8,8,8,c).permute(0,1,2,4,3,5,6).reshape(b,t,H,W,c)
        if p:
            x=x[:,:,p:p+h,p:p+w];mask=mask[:,:,p:p+h,p:p+w]
        z=x.permute(0,2,3,1,4).reshape(-1,t,c)
        m=mask.permute(0,2,3,1).reshape(-1,t)
        z=self.attend(self.time,self.norm[1],z,m)
        z=(z+self.ff(self.norm[2](z)))*m.unsqueeze(-1)
        return z.reshape(b,h,w,t,c).permute(0,3,1,2,4)

class FireViT(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.patch=nn.Conv2d(12,width,4,stride=4)
        self.position=nn.Parameter(torch.zeros(1,1,64,64,width))
        self.day=nn.Parameter(torch.zeros(1,4,1,1,width))
        nn.init.trunc_normal_(self.position,std=.02);nn.init.trunc_normal_(self.day,std=.02)
        self.blocks=nn.ModuleList(SpaceTimeBlock(width,o) for o in [0,4])
        self.head=nn.ModuleList(nn.Sequential(nn.Conv2d(width+12,16,3,padding=1),nn.GELU(),
                                               nn.Conv2d(16,1,1)) for _ in range(2))

    def forward(self,x,task):
        b,t,c,h,w=x.shape
        assert (t,c,h,w)==(4,12,256,256)
        flat=x.reshape(b*t,c,h,w)
        valid=F.max_pool2d(flat[:,-1:],4).reshape(b,t,64,64)>0
        z=self.patch(flat).permute(0,2,3,1).reshape(b,t,64,64,-1)
        z=(z+self.position+self.day)*valid.unsqueeze(-1)
        for block in self.blocks:z=block(z,valid)
        z=F.interpolate(z[:,-1].permute(0,3,1,2),size=(h,w),mode='bilinear',align_corners=False)
        return self.head[task](torch.cat([z,x[:,-1]],1)).squeeze(1)
