import csv,sys,glob
for t in sys.argv[1:]:
    f=glob.glob(f"/home/snehadri/bal_runs/{t}/*/lightning_logs/version_0/metrics.csv")
    if not f: print(f"### {t}: no metrics"); continue
    rows=list(csv.DictReader(open(f[0])))
    print(f"### {t}")
    for k in rows[0]:
        if 'balance' in k or k in ('train_loss','val_loss'):
            v=[float(r[k]) for r in rows if r.get(k) not in (None,'')]
            if v: print("   %-18s %9.4f -> %9.4f"%(k,v[0],v[-1]))
