import sys, json, numpy as np, torch, h5py, glob
sys.path.insert(0,"/home/snehadri/repos/unsupervised-search")
from model_blocks import Encoder, x_to_p4, ms_from_p4s, pairwise
REPO="/home/snehadri/repos/unsupervised-search"; H5="/home/snehadri/sim_software/output_h5s"
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
                if mod.bias is not None: mod.bias.zero_(); mod.eps=0.0
def build(ck):
    cfg=json.load(open(f"{REPO}/config_files/replication_config.json"))
    enc=Encoder(**cfg["model"]["encoder_config"])
    sd=torch.load(ck,map_location="cpu",weights_only=False)["state_dict"]
    enc.load_state_dict({k.replace("Encoder.",""):v for k,v in sd.items()}); enc.eval(); fuse(enc); return enc
@torch.no_grad()
def probe(enc,X):
    mask=(X[:,:,0]==0).bool(); w=torch.stack([X[:,:,1],X[:,:,2],X[:,:,3]],-1)
    loss,_,_,_=enc(X,w,mask)
    xx=enc.embed(X); xx=xx.masked_fill(mask.unsqueeze(-1).repeat(1,1,xx.shape[-1]).bool(),0)
    for ib in range(len(enc.obj_blocks)):
        if ib==0: wij=enc.mlp(pairwise(w)).squeeze(-1).repeat_interleave(enc.obj_blocks[ib].attn.num_heads,dim=0)
        xx=enc.obj_blocks[ib](Q=xx,K=xx,V=xx,key_padding_mask=mask.bool(),attn_mask=wij)
        xx=xx.masked_fill(mask.unsqueeze(-1).repeat(1,1,xx.shape[-1]).bool(),0)
        jc=enc.get_jet_choice(xx); c=torch.bmm(jc.transpose(2,1),xx)
        c=enc.cand_blocks[ib](Q=c,K=c,V=c,key_padding_mask=None,attn_mask=None)
        xx=enc.cross_blocks[ib](Q=xx,K=c,V=c,key_padding_mask=None,attn_mask=None)
        xx=xx.masked_fill(mask.unsqueeze(-1).repeat(1,1,xx.shape[-1]).bool(),0)
    jc=enc.get_jet_choice(xx)
    m=ms_from_p4s(torch.bmm(jc.transpose(2,1),x_to_p4(X))).numpy()[:,:2]
    a=jc.argmax(-1).numpy(); v=~mask.numpy()
    share=(np.minimum(((a==0)&v).sum(1),((a==1)&v).sum(1))/np.maximum(v.sum(1),1)).mean()
    isr=((a==2)&v).sum()/v.sum()
    return loss.numpy(), np.median(np.minimum(m[:,0],m[:,1])), np.median(np.maximum(m[:,0],m[:,1])), share, isr
def auc(b,s):
    x=np.concatenate([b,s]); r=x.argsort().argsort(); nb,ns=len(b),len(s)
    return (r[nb:].sum()-ns*(ns+1)/2)/(nb*ns)
N=1500
Xb=load(f"{H5}/qcd_background.h5",N)
sig={n:load(f"{H5}/{n}.h5",N) for n in ("gluino_rpv_6j","gluino_rpv_10j_fixed","squark_rpv_8j_2000")}
runs=[("deployed (retrained_noncollapse)",f"{REPO}/experiments/retrained_noncollapse/finalWeights.ckpt")]
for tag in ("balance10","moe5","gumbel0","gumbel10","gumbel50"):
    g=glob.glob(f"/home/snehadri/bal_runs/{tag}/*/finalWeights.ckpt")
    if g: runs.append((f"{tag} (2000 steps, seed 7)", g[0]))
print(f"{'model':<34}{'light share':>12}{'ISR':>7}{'m_light':>9}{'m_heavy':>9}   AUC 6j / 10j / 8j")
for name,ck in runs:
    enc=build(ck); lb,_,_,_,_=probe(enc,Xb)
    _,lo,hi,sh,isr=probe(enc,sig["gluino_rpv_6j"])
    aucs=[auc(lb,probe(enc,Xs)[0]) for Xs in sig.values()]
    print(f"{name:<34}{100*sh:>11.0f}%{100*isr:>6.0f}%{lo:>8.0f}G{hi:>8.0f}G   " + " / ".join(f"{a:.3f}" for a in aucs))
