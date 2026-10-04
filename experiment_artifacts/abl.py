import json, os, sys, time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
HOME=os.path.expanduser("~"); DATA=os.path.join(HOME,"nbaiot"); ST=os.path.join(HOME,"state5")
W=11
def win(z):
    n=len(z)//W; return np.median(z[:n*W].reshape(n,W),axis=1)
for dev in sys.argv[1:]:
    z=np.load(os.path.join(DATA,dev+".npz")); d={k:z[k] for k in z.files}
    rng=np.random.RandomState(42); ben=d["benign"].copy(); rng.shuffle(ben)
    n=len(ben); ntr,nva=int(.6*n),int(.2*n)
    sc=StandardScaler().fit(ben[:ntr]); Xtr=sc.transform(ben[:ntr]); Xva=sc.transform(ben[ntr:ntr+nva]); Xte=sc.transform(ben[ntr+nva:])
    m={"IF":IsolationForest(n_estimators=200,max_samples=1024,random_state=42,n_jobs=-1).fit(Xtr),
       "OCSVM":make_pipeline(Nystroem(gamma=0.1,n_components=100,random_state=42),SGDOneClassSVM(nu=0.05,random_state=42)).fit(Xtr),
       "AE":MLPRegressor(hidden_layer_sizes=(48,8,48),max_iter=120,random_state=42).fit(Xtr,Xtr)}
    def S(X): return np.column_stack([-m["IF"].decision_function(X),-m["OCSVM"].decision_function(X),
                                      np.mean((X-m["AE"].predict(X))**2,axis=1)])
    Sva=S(Xva); mu,sd=Sva.mean(0),Sva.std(0)+1e-9
    ys=[np.zeros(len(win(((S(Xte)-mu)/sd)[:,0])))]
    comp={k:[win(((S(Xte)-mu)/sd))[:,] ] for k in []}
    Ste=(S(Xte)-mu)/sd
    parts={"IF":[win(Ste[:,0])],"OCSVM":[win(Ste[:,1])],"AE":[win(Ste[:,2])],"ENS":[win(Ste.max(1))]}
    for lbl,X in sorted(d.items()):
        if lbl=="benign": continue
        Sa=(S(sc.transform(X))-mu)/sd
        parts["IF"].append(win(Sa[:,0])); parts["OCSVM"].append(win(Sa[:,1]))
        parts["AE"].append(win(Sa[:,2])); parts["ENS"].append(win(Sa.max(1)))
        ys.append(np.ones(len(win(Sa[:,0]))))
    y=np.concatenate(ys)
    out={k:round(float(roc_auc_score(y,np.concatenate(v))),4) for k,v in parts.items()}
    json.dump(out,open(os.path.join(ST,"abl_"+dev+".json"),"w"))
    print(dev,out,flush=True)
