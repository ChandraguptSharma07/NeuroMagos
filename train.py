import argparse, os, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import SynapseDataset
from model import HybridNeuroMagos, CNN1D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--wd", type=float, default=1e-4) # weight decay
p.add_argument("--resume", default=None)
p.add_argument("--save_dir", default=".")
p.add_argument("--model", default="resnet", choices=['resnet', 'cnn1d'])
args = p.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"running {args.model} on {dev}")

# dataset setup
dtype = 'raw' if args.model == 'cnn1d' else 'spec'
ds = SynapseDataset('Synapse_Dataset', mode='train', type=dtype)
tr = int(0.8 * len(ds))
train, val = random_split(ds, [tr, len(ds) - tr])

dl_tr = DataLoader(train, batch_size=args.bs, shuffle=True, num_workers=2)
dl_val = DataLoader(val, batch_size=args.bs, shuffle=False, num_workers=2)

# init model
if args.model == 'resnet':
    m = HybridNeuroMagos(8, 5).to(dev)
else:
    m = CNN1D(8, 5).to(dev)

if args.resume and os.path.exists(args.resume):
    print(f"resuming: {args.resume}")
    m.load_state_dict(torch.load(args.resume, map_location=dev))

opt = optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.wd)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

best_acc = 0
best_f = f"{args.save_dir}/best_{args.model}.pth"
last_f = f"{args.save_dir}/last_{args.model}.pth"
hist_f = f"{args.save_dir}/history_{args.model}.csv"
with open(hist_f, "w") as f: f.write("epoch,train_loss,val_acc\n")

for e in range(args.epochs):
    m.train()
    losses = []
    
    bar = tqdm(dl_tr, desc=f"Ep {e+1}")
    for x, y in bar:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = crit(m(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        bar.set_postfix(loss=np.mean(losses))
        
    # validate
    m.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for x, y in dl_val:
            x, y = x.to(dev), y.to(dev)
            out = m(x)
            cor += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
            
    acc = cor/tot
    print(f"Val: {acc:.4f}")
    sched.step() 
    
    # log history
    with open(f"{args.save_dir}/history_{args.model}.csv", "a") as f:
        f.write(f"{e+1},{np.mean(losses):.4f},{acc:.4f}\n")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(m.state_dict(), best_f)
        print("saved best")
    torch.save(m.state_dict(), last_f)

# report
print("generating report...")
preds, labs = [], []
m.eval()
with torch.no_grad():
    for x, y in dl_val:
        preds.extend(m(x.to(dev)).argmax(1).cpu().numpy())
        labs.extend(y.numpy())

rpt = classification_report(labs, preds, digits=4)
print(rpt)
with open(f"{args.save_dir}/report_{args.model}.txt", "w") as f: f.write(rpt)

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(labs, preds), annot=True, fmt='d', cmap='Blues')
plt.savefig(f"{args.save_dir}/conf_matrix_{args.model}.png")
print("done.")
