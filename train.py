import argparse, os, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import SynapseDataset
from model import HybridNeuroMagos
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# args
p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--wd", type=float, default=1e-4)
p.add_argument("--resume", type=str, default=None)
p.add_argument("--save_dir", type=str, default=".")
args = p.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"running on {dev}")

# data
ds = SynapseDataset('Synapse_Dataset', mode='train')
tr_len = int(0.8 * len(ds))
train_ds, val_ds = random_split(ds, [tr_len, len(ds) - tr_len])

train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2)

# model
model = HybridNeuroMagos(8, 5).to(dev)

if args.resume and os.path.exists(args.resume):
    print(f"loading ckpt: {args.resume}")
    model.load_state_dict(torch.load(args.resume, map_location=dev))

opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
crit = nn.CrossEntropyLoss()
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)

best = 0

for e in range(args.epochs):
    model.train()
    losses = []
    
    # train loop
    block = tqdm(train_dl, desc=f"Ep {e+1}")
    for x, y in block:
        x, y = x.to(dev), y.to(dev)
        
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        losses.append(loss.item())
        block.set_postfix(loss=np.mean(losses))
        
    # val loop
    model.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            cor += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
            
    acc = cor/tot
    print(f"Val Acc: {acc:.4f}")
    sched.step(acc)
    
    # save stuff
    if acc > best:
        best = acc
        torch.save(model.state_dict(), f"{args.save_dir}/best_model.pth")
        print("saved best")
        
    torch.save(model.state_dict(), f"{args.save_dir}/last_model.pth")

# final report
print("generating report...")
preds, labs = [], []
model.eval()
with torch.no_grad():
    for x, y in val_dl:
        x = x.to(dev)
        preds.extend(model(x).argmax(1).cpu().numpy())
        labs.extend(y.numpy())

rpt = classification_report(labs, preds, digits=4)
print(rpt)
with open(f"{args.save_dir}/classification_report.txt", "w") as f: f.write(rpt)

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(labs, preds), annot=True, fmt='d', cmap='Blues')
plt.savefig(f"{args.save_dir}/confusion_matrix.png")
print("done.")
