"""Label-free standalone prediction for geomfast checkpoints.
Only checkpoints, geometry, and optional priors are accepted; never Q3D labels.
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,importlib.util,time
from pathlib import Path
import numpy as np
import torch
W=Path(__file__).resolve().parent
from ..network import build_model
from ._tlsq.uge.scene import scene_from_gds,tlsq_namer
from ._tlsq.uge.graph import scene_graph
from ._tlsq.uge.blocks import edge_matrix
from ._tlsq.uge.proxies import CapacitanceMoM
from ._tlsq.uge import localtok as lt

class Predictor:
 def __init__(self,paths,threads=2):
  torch.set_num_threads(threads);self.models=[];self.threads=threads
  for path in paths:
   c=torch.load(path,map_location='cpu',weights_only=False)
   model=build_model(len(c['scales']['X']['median']),len(c['scales']['E']['median']),len(c['scales']['G']['median']),c['rounds']);model.load_state_dict(c['state']);model.eval();self.models.append((c,model))
  assert len({c['arm'] for c,m in self.models})==1
  self.arm=self.models[0][0]['arm'];self.direct=self.arm=='direct_gnn';self.fast=self.arm=='fast_gnn'
 def from_features(self,row,priors=None):
  n=len(row['names']);pairs=np.array([(i,j) for i in range(n) for j in range(i+1,n)]);i,j=pairs.T;assert len(row['edges'])==len(pairs),'complete graph required'
  outputs=[]
  for ck,model in self.models:
   schema=ck.get('schema',ck);nk=schema['nodekeys'];ek=schema['edgekeys'];gk=schema.get('globkeys',['log_chip_area','log_pocket_area','log_substrate_t','eps_eff'])
   X=np.array([[(float(not np.isfinite(nd['log_min_halfwidth'])) if k=='missing_log_min_halfwidth' else nd[k]) for k in nk] for nd in row['nodes']],np.float32)[None]
   E=edge_matrix(row['edges'],ek,n)[None];G=np.array([[row['global'][k] for k in gk]],np.float32)
   kinds=E[0,:,ek.index('is_ground_pair')].astype(int)
   if self.arm in ['pair_mlp','mesh','lofo_pair']:X=X[:,:,[nk.index('role_ground')]];G=np.zeros((1,1),np.float32)
   if self.direct:base=np.array([[ck['direct_intercepts'][str(k)] for k in kinds]],np.float32)
   else:
    p0,p1=(.75,1.5) if self.fast else (1.5,3.)
    lo=np.log(priors[p0][i,j]);hi=np.log(priors[p1][i,j]);assert np.isfinite(lo).all() and np.isfinite(hi).all()
    base=(2*hi-lo)[None].astype(np.float32);E=np.concatenate([E,(hi-lo)[None,:,None].astype(np.float32)],-1)
   for name,z in [('X',X),('E',E),('G',G)]:
    scale=ck['scales'][name];med=np.array(scale['median']);iqr=np.array(scale['iqr']);z[:]=np.clip((np.where(np.isfinite(z),z,med)-med)/iqr,-10,10)
    assert np.isfinite(z).all()
   closure=ck['closure_intercepts'];bbase=np.array([[closure.get('1',closure.get('True')) if nm!='G' else closure.get('0',closure.get('False')) for nm in row['names']]],np.float32)
   T=lambda a:torch.from_numpy(np.asarray(a));M=torch.ones((1,n),dtype=torch.bool);PM=torch.ones((1,len(pairs)),dtype=torch.bool)
   with torch.inference_mode():C,_=model(T(X),T(E),T(G),M,PM,T(pairs),T(base),T(bbase),self.direct)
   outputs.append(C[0].numpy())
  result=np.mean(outputs,axis=0);assert np.isfinite(result).all();return result
 def from_gds(self,path,names=None,alpha=None,local_gap=None):
  from ._tlsq.embed import proxy,fastprior
  physics=self.models[0][0].get('schema',{}).get('physics',{})
  if alpha is None:alpha=physics.get('alpha')
  if local_gap is None:local_gap=physics.get('prior_method')=='local_gap'
  proxy.NUM_THREADS=self.threads;start=time.perf_counter()
  if names is None:namer=tlsq_namer
  else:
   def namer(S,m):
    sig=[n for n in names if n!='G'];assert len(S)==len(sig),'signal polygons vs supplied file-order names';return sig
  sc=scene_from_gds(path,namer=namer);t1=time.perf_counter();row=lt.add_tokens(sc,scene_graph(sc));t2=time.perf_counter();priors={}
  if not self.direct:
   old=fastprior.ALPHA_GAP_SHARE
   try:
    if alpha is not None:fastprior.ALPHA_GAP_SHARE=alpha
    for p in ([.75,1.5] if self.fast else [1.5,3.]):
     if local_gap:
      from ._tlsq.embed.fastprior_loc import micro_mom_loc
      with proxy.substrate('slab_open',sc.stack['substrate_thickness_um']):
       mat,_,_=micro_mom_loc(sc.conductors[0].geom,[c.geom for c in sc.conductors[1:]],eps_r=sc.stack['substrate_eps_r'],ppg=p)
      priors[p]=-mat
     else:priors[p]=-CapacitanceMoM(max_cells=10**9,mesh='lite',cells_per_gap=p,substrate='slab_open').compute(sc,list(row['edges']))['matrix']
   finally:fastprior.ALPHA_GAP_SHARE=old
  t3=time.perf_counter();C=self.from_features(row,priors);t4=time.perf_counter()
  return dict(C=C,names=row['names'],timing=dict(scene=t1-start,geometry=t2-t1,prior=t3-t2,network_and_pack=t4-t3,total=t4-start))
