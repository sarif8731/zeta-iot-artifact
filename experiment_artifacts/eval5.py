#!/usr/bin/env python3
"""Final eval: per-device benign-only ensemble; per-instance AND windowed decisions."""
import glob, json, os, pickle, sys, time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

HOME=os.path.expanduser("~"); DATA=os.path.join(HOME,"nbaiot"); ST=os.path.join(HOME,"state5")
os.makedirs(ST, exist_ok=True)
W=11  # decision window (median over W consecutive snapshots)

def scores(m,X):
    return np.column_stack([-m["IF"].decision_function(X), -m["OCSVM"].decision_function(X),
                            np.mean((X-m["AE"].predict(X))**2,axis=1)])
def win_med(z,w=W):
    n=len(z)//w
    return np.median(z[:n*w].reshape(n,w),axis=1)

if sys.argv[1] != "finish":
  for dev in sys.argv[1:]:
    z=np.load(os.path.join(DATA,dev+".npz")); d={k:z[k] for k in z.files}
    rng=np.random.RandomState(42); ben=d["benign"].copy(); rng.shuffle(ben)
    n=len(ben); ntr,nva=int(.6*n),int(.2*n)
    scaler=StandardScaler().fit(ben[:ntr])
    Xtr,Xva,Xte=(scaler.transform(a) for a in (ben[:ntr],ben[ntr:ntr+nva],ben[ntr+nva:]))
    tt={}
    t0=time.time(); m={"IF":IsolationForest(n_estimators=200,max_samples=1024,random_state=42,n_jobs=-1).fit(Xtr)}; tt["IF"]=time.time()-t0
    t0=time.time(); m["OCSVM"]=make_pipeline(Nystroem(gamma=0.1,n_components=100,random_state=42),SGDOneClassSVM(nu=0.05,random_state=42)).fit(Xtr); tt["OCSVM"]=time.time()-t0
    t0=time.time(); m["AE"]=MLPRegressor(hidden_layer_sizes=(48,8,48),max_iter=120,random_state=42).fit(Xtr,Xtr); tt["AE"]=time.time()-t0
    Sva=scores(m,Xva); mu,sd=Sva.mean(0),Sva.std(0)+1e-9
    def fuse(X): return ((scores(m,X)-mu)/sd).max(1)
    zva=fuse(Xva)
    thr_i=float(np.quantile(zva,0.99))                    # instance-level
    wva=win_med(zva); thr_w=float(np.quantile(wva,0.99))  # window-level
    thr_hi=float(np.quantile(wva,0.999))
    out={"device":dev,"n_benign":n,"train_times_s":{k:round(v,2) for k,v in tt.items()},
         "model_MB":round(sum(len(pickle.dumps(x)) for x in m.values())/1e6,3),
         "labels":{}, "routing":{"ALLOW":0,"AUTONOMOUS":0,"ESCALATED":0,"DEFERRED":0},
         "thr_instance":round(thr_i,3),"thr_window":round(thr_w,3)}
    crit=any(k in dev.lower() for k in ("camera","doorbell","monitor","webcam"))
    zte=fuse(Xte); wte=win_med(zte)
    out["labels"]["benign"]={"n":len(zte),"n_win":len(wte),
        "det_inst":round(float((zte>thr_i).mean()),4),"det_win":round(float((wte>thr_w).mean()),4)}
    all_w,all_y=[wte],[np.zeros(len(wte))]
    for lbl,X in sorted(d.items()):
        if lbl=="benign": continue
        za=fuse(scaler.transform(X)); wa=win_med(za)
        out["labels"][lbl]={"n":len(za),"n_win":len(wa),
            "det_inst":round(float((za>thr_i).mean()),4),"det_win":round(float((wa>thr_w).mean()),4)}
        all_w.append(wa); all_y.append(np.ones(len(wa)))
    wall,yall=np.concatenate(all_w),np.concatenate(all_y)
    out["auc_window"]=round(float(roc_auc_score(yall,wall)),4)
    for v in wall:
        if v<=thr_w: out["routing"]["ALLOW"]+=1
        elif v>=thr_hi: out["routing"]["DEFERRED" if crit and v>3*thr_hi else ("ESCALATED" if crit else "AUTONOMOUS")]+=1
        else: out["routing"]["ESCALATED"]+=1
    X1=Xte[:1]; t0=time.time()
    for _ in range(50): fuse(X1)
    out["latency_single_ms"]=round((time.time()-t0)/50*1000,2)
    Xb=np.vstack([Xte]*2)[:4096]; t0=time.time(); fuse(Xb)
    out["latency_batch_ms"]=round((time.time()-t0)/len(Xb)*1000,4)
    json.dump(out,open(os.path.join(ST,dev+".json"),"w"),indent=1)
    print(dev,"FPRw",out["labels"]["benign"]["det_win"],"AUCw",out["auc_window"],flush=True)
else:
    per_dev,fam_i,fam_w,routing={},{},{},{"ALLOW":0,"AUTONOMOUS":0,"ESCALATED":0,"DEFERRED":0}
    nA=nAi=nAw=nB=nBi=nBw=0; aucs=[];lat1=[];latb=[];mbs=[];tts=[]
    nAiN=nAwN=nBiN=nBwN=0
    for f in sorted(glob.glob(os.path.join(ST,"*.json"))):
        if "FINAL" in f: continue
        d=json.load(open(f)); per_dev[d["device"]]=d["labels"]
        aucs.append(d["auc_window"]);lat1.append(d["latency_single_ms"]);latb.append(d["latency_batch_ms"]);mbs.append(d["model_MB"]);tts.append(sum(d["train_times_s"].values()))
        for k,v in d["routing"].items(): routing[k]+=v
        for lbl,r in d["labels"].items():
            if lbl=="benign":
                nBi+=r["n"]*r["det_inst"]; nBiN+=r["n"]; nBw+=r["n_win"]*r["det_win"]; nBwN+=r["n_win"]
            else:
                nAi+=r["n"]*r["det_inst"]; nAiN+=r["n"]; nAw+=r["n_win"]*r["det_win"]; nAwN+=r["n_win"]
                fam_i.setdefault(lbl,[]).append(r["det_inst"]); fam_w.setdefault(lbl,[]).append(r["det_win"])
    gaf=[x for k,v in fam_w.items() if k.startswith("gafgyt") for x in v]
    mir=[x for k,v in fam_w.items() if k.startswith("mirai") for x in v]
    tot_r=sum(routing.values())
    res={"protocol":"per-device benign-only training (60/20/20), IF(200,1024)+Nystroem-SGDOCSVM+AE(48-8-48,120it), z-max fusion, decision window W=11 (median), thresholds 99th pct benign validation",
      "window_W":W,"n_devices":len(per_dev),
      "instance_level":{"TDR":round(nAi/nAiN,4),"FPR":round(nBi/nBiN,4)},
      "window_level":{"TDR":round(nAw/nAwN,4),"FPR":round(nBw/nBwN,4),
        "TDR_gafgyt":round(float(np.mean(gaf)),4),"TDR_mirai":round(float(np.mean(mir)),4),
        "macro_AUC":round(float(np.mean(aucs)),4),
        "n_test_attack_windows":int(nAwN),"n_test_benign_windows":int(nBwN)},
      "resources":{"latency_single_ms_mean":round(float(np.mean(lat1)),2),
        "latency_batch_ms_mean":round(float(np.mean(latb)),4),
        "model_MB_mean":round(float(np.mean(mbs)),2),"train_time_s_mean":round(float(np.mean(tts)),2)},
      "per_attack_TDR_window":{k:round(float(np.mean(v)),4) for k,v in sorted(fam_w.items())},
      "per_attack_TDR_instance":{k:round(float(np.mean(v)),4) for k,v in sorted(fam_i.items())},
      "per_device":per_dev,
      "taige_routing_windows":{"counts":routing,"fractions":{k:round(v/tot_r,4) for k,v in routing.items()}}}
    json.dump(res,open(os.path.join(ST,"FINAL_RESULTS.json"),"w"),indent=1)
    print(json.dumps({k:res[k] for k in ["instance_level","window_level","resources","per_attack_TDR_window","taige_routing_windows"]},indent=1))
