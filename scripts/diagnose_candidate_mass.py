#!/usr/bin/env python3
"""Why are the reconstructed candidate masses far below the sparticle mass?

Runs the float PyTorch model and reports, at each point where a jet choice can
be taken, the resulting candidate masses, the jets per candidate, how often the
assignment collapses (every jet in one category) and the light candidate's share.

Usage: diagnose_candidate_mass.py [sample] [n_events]   (CKPT= to override)
Findings are written up in figs/candidate_mass_diagnosis.txt.
"""
import sys, json, numpy as np, torch, h5py
sys.path.insert(0,"/home/snehadri/repos/unsupervised-search")
from model_blocks import Encoder, x_to_p4, ms_from_p4s
REPO="/home/snehadri/repos/unsupervised-search"
import os
CKPT=os.environ.get("CKPT", f"{REPO}/experiments/retrained_noncollapse/finalWeights.ckpt")
H5="/home/snehadri/sim_software/output_h5s"
SAMPLE=sys.argv[1] if len(sys.argv)>1 else "gluino_rpv_6j"
NEV=int(sys.argv[2]) if len(sys.argv)>2 else 2000
def load(fn,n):
    with h5py.File(fn,"r") as f:
        e=np.nan_to_num(np.array(f["source"]["e"]))/1000.; pt=np.nan_to_num(np.array(f["source"]["pt"]))/1000.
        with np.errstate(divide="ignore"):
            le=np.log(e); le[~np.isfinite(le)]=0; lp=np.log(pt); lp[~np.isfinite(lp)]=0
        phi=np.array(f["source"]["phi"]); eta=np.array(f["source"]["eta"])
        X=np.stack([lp,eta,np.cos(phi),np.sin(phi),le],-1); X=X[(pt>0).sum(1)>=6]
    return torch.tensor(X[:n],dtype=torch.float32)
def fuse(m):
    for mod in m.modules():
        if isinstance(mod,torch.nn.BatchNorm1d):
            with torch.no_grad():
                mod.running_mean.zero_(); mod.running_var.fill_(1.0)
                if mod.weight is not None: mod.weight.fill_(1.0)
                if mod.bias is not None: mod.bias.zero_()
                mod.eps=0.0
cfg=json.load(open(f"{REPO}/config_files/replication_config.json"))
enc=Encoder(**cfg["model"]["encoder_config"])
sd=torch.load(CKPT,map_location="cpu",weights_only=False)["state_dict"]
enc.load_state_dict({k.replace("Encoder.",""):v for k,v in sd.items()})
enc.eval(); fuse(enc)
X=load(f"{H5}/{SAMPLE}.h5",NEV)
mask=(X[:,:,0]==0).bool(); w=torch.stack([X[:,:,1],X[:,:,2],X[:,:,3]],-1)

@torch.no_grad()
def probe(X,w,mask):
    originalx=X; x=enc.embed(X); x=x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(),0)
    out={}
    for ib in range(len(enc.obj_blocks)):
        if enc.doWij and ib==0:
            from model_blocks import pairwise
            wij=enc.mlp(pairwise(w)).squeeze(-1).repeat_interleave(enc.obj_blocks[ib].attn.num_heads,dim=0)
        x=enc.obj_blocks[ib](Q=x,K=x,V=x,key_padding_mask=mask.bool(),attn_mask=wij)
        x=x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(),0)
        jc=enc.get_jet_choice(x); out[f"jc_obj{ib}"]=jc
        c=torch.bmm(jc.transpose(2,1),x)
        c=enc.cand_blocks[ib](Q=c,K=c,V=c,key_padding_mask=None,attn_mask=None)
        x=enc.cross_blocks[ib](Q=x,K=c,V=c,key_padding_mask=None,attn_mask=None)
        x=x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(),0)
        out[f"jc_cross{ib}"]=enc.get_jet_choice(x)
    out["jp4"]=x_to_p4(originalx)
    return out
o=probe(X,w,mask)
jp4=o["jp4"]
print(f"sample {SAMPLE}, {len(X)} events, ckpt {CKPT.split(chr(47))[-2]}")
# candidate 0/1 labels are arbitrary per event, so sort each event's pair
nj_real=(~mask.numpy()).sum(1)
print(f"real jets/event: mean {nj_real.mean():.1f}")
print(f"{'jet choice after':<20}{'m_light':>10}{'m_heavy':>10}{'jets:c0':>9}{'jets:c1':>9}{'jets:ISR':>9}{'collapsed':>10}{'light%':>9}")
for k in ("jc_obj0","jc_cross0","jc_obj1","jc_cross1"):
    jc=o[k]; p4=torch.bmm(jc.transpose(2,1),jp4); m=ms_from_p4s(p4).numpy()[:,:2]
    lo=np.minimum(m[:,0],m[:,1]); hi=np.maximum(m[:,0],m[:,1])
    a=jc.argmax(-1).numpy(); valid=~mask.numpy()
    n0=((a==0)&valid).sum(1).mean(); n1=((a==1)&valid).sum(1).mean(); n2=((a==2)&valid).sum(1).mean()
    # collapse = every real jet in the event lands in one category
    per_ev=[]
    for e in range(len(a)):
        v=a[e][valid[e]]
        per_ev.append(len(np.unique(v))==1 if len(v) else True)
    coll=100*np.mean(per_ev)
    frac_light=(np.minimum(((a==0)&valid).sum(1),((a==1)&valid).sum(1))/np.maximum(valid.sum(1),1)).mean()
    print(f"{k:<20}{np.median(lo):>9.0f}G{np.median(hi):>9.0f}G{n0:>9.1f}{n1:>9.1f}{n2:>9.1f}{coll:>9.0f}%{100*frac_light:>9.0f}%")
