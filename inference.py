import torch, pandas as pd, argparse, os
from model import HybridNeuroMagos
import preprocessing as pp

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--model", default="best_model.pth")
args = p.parse_args()

def process(f):
    try:
        raw = pd.read_csv(f).iloc[:, :8].values
        d = pp.notch(raw)
        d = pp.bandpass(d)
        d = pp.norm(d)
        s = pp.get_spec(d)
        return torch.tensor(s, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        print(e)
        return None

# load
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"loading {args.model}")
model = HybridNeuroMagos(8, 5).to(dev)
model.load_state_dict(torch.load(args.model, map_location=dev))
model.eval()

# run
pt = process(args.input)
if pt is not None:
    with torch.no_grad():
        out = model(pt.to(dev))
        res = out.argmax(1).item()
        
    print(f"File: {args.input} -> Gesture: {res}")
