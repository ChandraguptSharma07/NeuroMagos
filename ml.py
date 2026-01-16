import glob, os, numpy as np, pandas as pd, joblib
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import preprocessing as pp

def feats(d):
    # d: [8, T]
    f = []
    for c in d:
        f.extend([np.mean(c), np.std(c), np.min(c), np.max(c), 
                  np.sqrt(np.mean(c**2)), skew(c), kurtosis(c)])
    return np.array(f)

print("loading...")
fs = glob.glob("Synapse_Dataset/**/gesture*.csv", recursive=True)
X, y = [], []

for f in fs:
    try:
        raw = pd.read_csv(f).iloc[:,:8].values
        # use same prep as dl
        d = pp.norm(pp.bandpass(pp.notch(raw))).T
        X.append(feats(d))
        y.append(int(os.path.basename(f).split('_')[0].replace('gesture','')))
    except: pass

X, y = np.array(X), np.array(y)
print(f"got {X.shape} {y.shape}")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
mdls = {
    'xgb': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'rf': RandomForestClassifier(n_estimators=100),
    'svm': SVC(probability=True)
}

res = {k:[] for k in mdls}

print("training fold...")
for i, (ti, vi) in enumerate(skf.split(X, y)):
    xt, xv = X[ti], X[vi]
    yt, yv = y[ti], y[vi]
    
    for k, m in mdls.items():
        m.fit(xt, yt)
        acc = accuracy_score(yv, m.predict(xv))
        res[k].append(acc)
        
    print(f"f{i}: xgb={res['xgb'][-1]:.3f} rf={res['rf'][-1]:.3f} svm={res['svm'][-1]:.3f}")
    
    # save fold 0 just in case
    if i == 0:
        for k, m in mdls.items(): joblib.dump(m, f"ml_{k}_f0.pkl")

print("\nstats:")
for k, v in res.items():
    print(f"{k}: {np.mean(v):.4f} +/- {np.std(v):.4f}")

print("retraining full...")
for k, m in mdls.items():
    m.fit(X, y)
    joblib.dump(m, f"ml_{k}_full.pkl")
print("done.")
