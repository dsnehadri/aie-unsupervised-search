"""Pack inputs AND compute the PyTorch reference from the SAME array, so the
C++ harness and the reference cannot drift apart."""
import sys, json, struct, numpy as np, torch, h5py
sys.path.insert(0, "/home/snehadri/repos/unsupervised-search")
from model import StepLightning
from model_blocks import pairwise
from export_phase3 import fuse_all_batchnorms

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
S = "/home/snehadri/aie_scratch_save_20260810"

with h5py.File("/home/snehadri/repos/unsupervised-search/inputs/qcd_background.h5") as f:
    e = np.nan_to_num(np.array(f['source']['e'])) / 1000.
    pt = np.nan_to_num(np.array(f['source']['pt'])) / 1000.
    with np.errstate(divide='ignore'):
        le = np.log(e);  le[~np.isfinite(le)] = 0
        lp = np.log(pt); lp[~np.isfinite(lp)] = 0
    phi = np.array(f['source']['phi']); eta = np.array(f['source']['eta'])
    X = np.stack([lp, eta, np.cos(phi), np.sin(phi), le], -1)
    X = X[(pt > 0).sum(1) >= 6][:N]

ev = np.zeros((X.shape[0], 12, 5), np.float32)
k = min(12, X.shape[1]); ev[:, :k, :] = X[:, :k, :]
mask_np = (ev[:, :, 0] == 0)

words = []
for i in range(len(ev)):
    words += [int(w) for w in ev[i].reshape(-1).view(np.uint32)]
    words += [1 if x else 0 for x in mask_np[i]]
open(f"{S}/matched{N}.bin", "wb").write(struct.pack(f"{len(words)}I", *words))

cfg = json.load(open("/home/snehadri/repos/unsupervised-search/config_files/replication_config.json"))
model = StepLightning(**cfg["model"])
ck = torch.load("/home/snehadri/repos/unsupervised-search/experiments/retrained_noncollapse/finalWeights.ckpt",
                map_location="cpu", weights_only=False)
model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
enc = model.encoder if hasattr(model, "encoder") else model.Encoder
enc.eval(); fuse_all_batchnorms(enc); enc.eval()

xb = torch.tensor(ev, dtype=torch.float32)
with torch.no_grad():
    mask = (xb[:, :, 0] == 0).bool()
    w = torch.stack([xb[:, :, 1], xb[:, :, 2], xb[:, :, 3]], -1)
    x = enc.embed(xb)
    x = x.masked_fill(mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]).bool(), 0)
    emb = x.clone()
    wij = enc.mlp(pairwise(w)).squeeze(-1)
    wij_exp = wij.repeat_interleave(enc.obj_blocks[0].attn.num_heads, dim=0)
    blk = enc.obj_blocks[0]
    resid = x.clone()
    a, _ = blk.attn(query=x, key=x, value=x, key_padding_mask=mask.bool(), attn_mask=wij_exp)
    if a.shape == resid.shape: a = a + resid
    a = blk.post_attn_norm(a)
    x = blk.ffwd(a) + a
    x = blk.post_ffwd_norm(x)
    x = x.masked_fill(mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]).bool(), 0)

np.save(f"{S}/m_obj{N}.npy", x.numpy())
np.save(f"{S}/m_embed{N}.npy", emb.numpy())
np.save(f"{S}/m_mask{N}.npy", mask.numpy())
print(f"matched set: {len(ev)} events -> matched{N}.bin + m_obj{N}.npy")
