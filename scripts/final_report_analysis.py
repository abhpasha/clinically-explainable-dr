"""Reproduce the main report-level analyses from prediction CSVs.

Example:
python scripts/final_report_analysis.py \
  --validation-predictions outputs/validation/predictions.csv \
  --test-predictions outputs/final_test/predictions.csv \
  --fp-csv /path/to/MMRDR-CFP/FP.csv \
  --output outputs/final_report_analysis

This script does not train models and does not select a new model from the test set.
"""
from __future__ import annotations
import argparse, ast, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, log_loss


CLASSES=["No DR","Mild NPDR","Moderate NPDR","Severe NPDR","PDR"]
LESIONS=[
    "Microaneurysm","Hard exudate","Intraretinal haemorrhage",
    "Venous beading or IRMA","Neovascularisation",
    "Vitreous haemorrhage","Retinal detachment"
]


def metrics(y,p):
    return {
        "accuracy":float(accuracy_score(y,p)),
        "macro_f1":float(f1_score(y,p,labels=range(5),average="macro",zero_division=0)),
        "qwk":float(cohen_kappa_score(y,p,labels=range(5),weights="quadratic")),
    }


def bootstrap_ci(y,p,seed=42,reps=2000):
    obs=metrics(y,p); rng=np.random.default_rng(seed); store={k:[] for k in obs}
    for _ in range(reps):
        idx=rng.integers(0,len(y),len(y)); m=metrics(y[idx],p[idx])
        for k,v in m.items(): store[k].append(v)
    rows=[]
    for k,v in obs.items():
        lo,hi=np.quantile(store[k],[.025,.975])
        rows.append({"metric":k,"estimate":v,"ci_low":float(lo),"ci_high":float(hi)})
    return pd.DataFrame(rows)


def clinical_errors(y,p):
    return {
        "n":int(len(y)),"errors":int((p!=y).sum()),
        "undergraded":int((p<y).sum()),
        "undergraded_by_2plus":int(((y-p)>=2).sum()),
        "pdr_undergraded":int(((y==4)&(p<4)).sum()),
        "dr_grade1plus_predicted_no_dr":int(((y>=1)&(p==0)).sum()),
        "normal_upgraded":int(((y==0)&(p>0)).sum()),
    }


def wilson(k,n,z=1.959963984540054):
    if not n:return (np.nan,np.nan)
    q=k/n; den=1+z*z/n; c=(q+z*z/(2*n))/den
    h=z*np.sqrt((q*(1-q)+z*z/(4*n))/n)/den
    return float(c-h),float(c+h)


def grade2plus(y,p):
    t=y>=2; q=p>=2
    tp=int((t&q).sum()); fn=int((t&~q).sum()); tn=int((~t&~q).sum()); fp=int((~t&q).sum())
    s=tp/(tp+fn); c=tn/(tn+fp); slo,shi=wilson(tp,tp+fn); clo,chi=wilson(tn,tn+fp)
    return {
        "sensitivity":s,"sensitivity_ci":[slo,shi],
        "specificity":c,"specificity_ci":[clo,chi],
        "tp":tp,"fn":fn,"tn":tn,"fp":fp,
        "note":"Exploratory grade >=2 threshold; not an NHS referral rule."
    }


def calibration(y,probs):
    conf=probs.max(1); pred=probs.argmax(1); one=np.eye(5)[y]
    brier=float(np.mean(np.sum((probs-one)**2,axis=1)))
    nll=float(log_loss(y,probs,labels=range(5)))
    edges=np.linspace(0,1,11); ids=np.clip(np.digitize(conf,edges)-1,0,9); ece=0.
    for b in range(10):
        m=ids==b
        if m.any(): ece+=m.mean()*abs(float(conf[m].mean())-float((pred[m]==y[m]).mean()))
    return {"ece_10_bins":float(ece),"multiclass_brier_score":brier,"negative_log_likelihood":nll}


def parse_lesion(v):
    try:
        a=ast.literal_eval(str(v))
        if not isinstance(a,(list,tuple)):a=[a]
    except Exception:
        a=[int(x) for x in re.findall(r"(?<!\d)[01](?!\d)",str(v))]
    if len(a)!=7: raise ValueError(f"Could not parse lesion vector: {v!r}")
    return [int(float(x)) for x in a]


def mild_pattern(pred,fp):
    d=pred.copy(); d["image"]=d["path"].map(lambda x:Path(str(x)).name)
    a=fp[["image","grade","lesion"]].copy(); a["image"]=a.image.map(lambda x:Path(str(x)).name)
    lm=pd.DataFrame(a.lesion.map(parse_lesion).tolist(),columns=LESIONS,index=a.index)
    a=pd.concat([a[["image","grade"]],lm],axis=1)
    d=d.merge(a,on="image",how="left"); m=d[d.true_grade==1].copy(); other=LESIONS[1:]
    m["pattern"]=np.where(
        (m["Microaneurysm"]==1)&(m[other].max(1)==0),"MA only",
        np.where((m["Microaneurysm"]==1)&(m[other].max(1)==1),"MA + other lesion(s)","Other")
    )
    rows=[]
    for name,g in m.groupby("pattern"):
        rows.append({
            "pattern":name,"n":int(len(g)),
            "mild_recall":float((g.pred_grade==1).mean()),
            "predicted_no_dr_share":float((g.pred_grade==0).mean()),
            "predicted_moderate_share":float((g.pred_grade==2).mean()),
        })
    return pd.DataFrame(rows)


def run_prediction_analysis(path,out,seed):
    d=pd.read_csv(path); y=d.true_grade.to_numpy(int); p=d.pred_grade.to_numpy(int)
    probs=d[[f"prob_{i}" for i in range(5)]].to_numpy(float)
    bootstrap_ci(y,p,seed).to_csv(out/"bootstrap_ci.csv",index=False)
    (out/"clinical_errors.json").write_text(json.dumps(clinical_errors(y,p),indent=2))
    (out/"grade2plus.json").write_text(json.dumps(grade2plus(y,p),indent=2))
    (out/"calibration.json").write_text(json.dumps(calibration(y,probs),indent=2))
    return d


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--validation-predictions",required=True)
    p.add_argument("--test-predictions",required=True)
    p.add_argument("--fp-csv",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    vo=out/"validation"; to=out/"test"; vo.mkdir(exist_ok=True); to.mkdir(exist_ok=True)
    v=run_prediction_analysis(a.validation_predictions,vo,a.seed)
    t=run_prediction_analysis(a.test_predictions,to,a.seed+999)
    fp=pd.read_csv(a.fp_csv)
    mild_pattern(v,fp).to_csv(vo/"mild_npdr_lesion_pattern.csv",index=False)
    mild_pattern(t,fp).to_csv(to/"mild_npdr_lesion_pattern.csv",index=False)
    print(f"Saved report analyses to {out}")


if __name__=="__main__":
    main()
