import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,d,h,hidden):
        super().__init__(); self.h=h; self.n1=nn.RMSNorm(d); self.n2=nn.RMSNorm(d)
        self.qkv=nn.Linear(d,3*d,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.gate=nn.Linear(d,hidden,bias=False); self.up=nn.Linear(d,hidden,bias=False); self.down=nn.Linear(hidden,d,bias=False)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.n1(x)).view(b,t,3,self.h,d//self.h).permute(2,0,3,1,4).unbind(0)
        freq=1/(10000**(torch.arange(0,d//self.h,2,device=x.device).float()/(d//self.h)))
        angle=torch.outer(torch.arange(t,device=x.device),freq); c=angle.cos().to(q.dtype); s=angle.sin().to(q.dtype)
        def rope(z):
            a,b=z[...,::2],z[...,1::2]
            return torch.stack((a*c-b*s,a*s+b*c),dim=-1).flatten(-2)
        x=x+self.o(F.scaled_dot_product_attention(rope(q),rope(k),v,is_causal=True).transpose(1,2).reshape(b,t,d))
        z=self.n2(x); return x+self.down(F.silu(self.gate(z))*self.up(z))

class LM(nn.Module):
    def __init__(self,vocab=32000,d=768,layers=12,heads=12,hidden=1792):
        super().__init__(); self.emb=nn.Embedding(vocab,d); self.blocks=nn.ModuleList([Block(d,heads,hidden) for _ in range(layers)]); self.norm=nn.RMSNorm(d)
        self.apply(self.init)
    def init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,std=.02)
    def forward(self,x,y=None):
        z=self.emb(x)
        for block in self.blocks: z=block(z)
        z=self.norm(z)
        if y is None: return F.linear(z,self.emb.weight)
        # 分块计算词表 logits，降低峰值显存；checkpoint 避免保留全部 logits。
        from torch.utils.checkpoint import checkpoint
        z=z.flatten(0,1); y=y.flatten(); loss=z.new_zeros((),dtype=torch.float32)
        def ce(a,b): return F.cross_entropy(F.linear(a,self.emb.weight).float(),b,reduction='sum')
        for start in range(0,len(y),256):
            a,b=z[start:start+256],y[start:start+256]
            loss=loss+(checkpoint(ce,a,b,use_reentrant=False) if self.training else ce(a,b))
        return loss/(y != -100).sum().clamp_min(1)

