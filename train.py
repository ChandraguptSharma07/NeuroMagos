import argparse, os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from dataset import SynapseDataset
from model import HybridNeuroMagos, CNN1D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--wd", type=float, default=1e-4)
p.add_argument("--resume", default=None)
p.add_argument("--save_dir", default=".")
p.add_argument("--model", default="resnet", choices=['resnet', 'cnn1d'])
p.add_argument("--mixup", type=int, default=1)
p.add_argument("--swa", type=int, default=1)
p.add_argument("--fold", type=int, default=0)
p.add_argument("--k_folds", type=int, default=5)
p.add_argument("--tta", type=int, default=1)
args = p.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"run {args.model} | fold {args.fold}/{args.k_folds} | tta={args.tta} | mix={args.mixup} | swa={args.swa}")

dtype = 'raw' if args.model == 'cnn1d' else 'spec'
ds = SynapseDataset('Synapse_Dataset', mode='train', type=dtype)

kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
splits = list(kf.split(range(len(ds))))
tr_idx, val_idx = splits[args.fold]
train, val = Subset(ds, tr_idx), Subset(ds, val_idx)

dl_tr = DataLoader(train, batch_size=args.bs, shuffle=True, num_workers=2)
dl_val = DataLoader(val, batch_size=args.bs, shuffle=False, num_workers=2)

if args.model == 'resnet': m = HybridNeuroMagos(8, 5).to(dev)
else: m = CNN1D(8, 5).to(dev)

if args.resume and os.path.exists(args.resume):
    print(f"resuming {args.resume}")
    m.load_state_dict(torch.load(args.resume, map_location=dev))

opt = optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.wd)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)

swa_m = AveragedModel(m)
swa_sch = SWALR(opt, swa_lr=0.05)
swa_start = int(args.epochs * 0.75)

best_acc = 0
best_f = f"{args.save_dir}/best_{args.model}_f{args.fold}.pth"
hist_f = f"{args.save_dir}/hist_{args.model}_f{args.fold}.csv"
with open(hist_f, "w") as f: f.write("ep,loss,acc\n")

for e in range(args.epochs):
    m.train()
    losses = []
    bar = tqdm(dl_tr, desc=f"Ep {e+1}")
    for x, y in bar:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        if args.mixup:
            lam = np.random.beta(1.0, 1.0)
            idx = torch.randperm(x.size(0)).to(dev)
            mix = lam * x + (1 - lam) * x[idx]
            ya, yb = y, y[idx]
            out = m(mix)
            loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb)
        else:
            loss = crit(m(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        bar.set_postfix(loss=np.mean(losses))
    
    # eval
    eval_m = swa_m if (args.swa and e > swa_start) else m
    if args.swa and e > swa_start: update_bn(dl_tr, swa_m, device=dev)
    eval_m.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for x, y in dl_val:
            x, y = x.to(dev), y.to(dev)
            out = eval_m(x)
            cor += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
    acc = cor/tot
    print(f"Val: {acc:.4f}")
    with open(hist_f, "a") as f: f.write(f"{e+1},{np.mean(losses):.4f},{acc:.4f}\n")
    if acc > best_acc:
        best_acc = acc
        torch.save(eval_m.state_dict(), best_f)
        print("saved best")

    if args.swa and e > swa_start:
        swa_m.update_parameters(m)
        swa_sch.step()
    else: sched.step(acc)

# final TTA eval
print("TTA eval...")
fin_m = swa_m if args.swa else m
if args.swa: update_bn(dl_tr, swa_m, device=dev)
fin_m.eval()
preds, labs = [], []
with torch.no_grad():
    for x, y in dl_val:
        x = x.to(dev)
        if args.tta:
            # avg prediction with/without noise
            o1 = fin_m(x).softmax(1)
            o2 = fin_m(x + torch.randn_like(x)*0.01).softmax(1)
            p = (o1 + o2) / 2
        else: p = fin_m(x)
        preds.extend(p.argmax(1).cpu().numpy())
        labs.extend(y.numpy())

rpt = classification_report(labs, preds, digits=4)
print(rpt)
with open(f"{args.save_dir}/report_{args.model}_f{args.fold}.txt", "w") as f: f.write(rpt)
plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(labs, preds), annot=True, fmt='d', cmap='Blues')
plt.savefig(f"{args.save_dir}/conf_{args.model}_f{args.fold}.png")
print("done.")
