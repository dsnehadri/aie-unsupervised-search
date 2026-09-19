"""Where the data lives.

DATA holds every input the figure scripts read and every cache they write:
autoencoder-loss and anomaly-baseline caches, the weighted background, block
and tile sweeps, the validation dumps, and the link-benchmark sweeps. It is
`data/` in this repository; set PASSWD_DATA to point somewhere else.

MODEL_REPO is the upstream Passwd-ABC repository (Ref. [20] in the paper). Only
the scripts that need the trained checkpoint or its batcher use it; the figure
scripts do not.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("PASSWD_DATA", os.path.join(REPO, "data"))
FIGS = os.path.join(REPO, "figs")
MODEL_REPO = os.environ.get("PASSWD_MODEL_REPO", "/home/snehadri/repos/unsupervised-search")
