# ruff: noqa
# fmt: off
"""Networks of the planar-capacitance models (inference and training share these definitions).

``build_model``: the residual message-passing network of QCap and QCap-Fast (``full_gnn``; also
``direct_gnn`` / ``full_mlp`` / ``pair_mlp`` checkpoints), verbatim from tl-squadds ``scripts/geomfast/pilot_p2.py``
(sha256 in ``_core/SOURCES.json`` under ``network.py``). ``build_surrogate``: the learned prior surrogate of QCap-Fast
(``_core/lp_surrogate.py`` ``TriGNN``).
"""


def build_model(dx,de,dg,rounds,width=64):
 import torch
 from torch import nn
 def mlp(i,o):return nn.Sequential(nn.Linear(i,width),nn.SiLU(),nn.Linear(width,o))
 class Net(nn.Module):
  def __init__(self):
   super().__init__(); self.node=mlp(dx+dg,width); self.edge=mlp(de,width)
   self.msg=nn.ModuleList([mlp(3*width,width) for _ in range(rounds)])
   self.up=nn.ModuleList([mlp(2*width,width) for _ in range(rounds)])
   self.head=mlp(3*width,1); self.closure=mlp(width,1)
   nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
   nn.init.zeros_(self.closure[-1].weight); nn.init.zeros_(self.closure[-1].bias)
  def forward(self,X,E,G,M,PM,pairs,base,bbase,direct=False):
   B,N,_=X.shape; i,j=pairs[:,0],pairs[:,1]
   h=self.node(torch.cat([X,G[:,None].expand(-1,N,-1)],-1))*M[:,:,None]; e=self.edge(E)
   for msg,up in zip(self.msg,self.up):
    v=msg(torch.cat([h[:,i]+h[:,j],abs(h[:,i]-h[:,j]),e],-1))*PM[:,:,None]
    agg=torch.zeros_like(h); agg.index_add_(1,i,v); agg.index_add_(1,j,v)
    degree=torch.zeros_like(M,dtype=X.dtype); degree.index_add_(1,i,PM.float()); degree.index_add_(1,j,PM.float())
    h=(h+up(torch.cat([h,agg/degree.clamp_min(1)[:,:,None]],-1)))*M[:,:,None]
   raw=self.head(torch.cat([h[:,i]+h[:,j],abs(h[:,i]-h[:,j]),e],-1)).squeeze(-1)
   logc=(base+(raw if direct else .2*torch.tanh(raw))).clamp(-25,20)
   mutual=logc.exp()*PM; b=(bbase+self.closure(h).squeeze(-1)).clamp(-20,20).exp()*M
   C=torch.zeros((B,N,N),device=X.device); C[:,i,j]=-mutual; C[:,j,i]=-mutual
   diag=-C.sum(-1)+b; C=C+torch.diag_embed(diag)
   return C,logc
 return Net()


# fmt: on


def build_surrogate(dx, de, dg, w=128, rounds=4):
    from ._core.lp_surrogate import build

    return build(dx, de, dg, w, rounds)
