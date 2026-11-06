"""Independent source-cohort accounting plus exact checkpoint selection replay.

Grid dynamics/reward/probability accounting is independently reconstructed.
Selection replay uses the reviewed learner implementation: it verifies saved
predictions and updates, not independent scientific replication or target GO.
"""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from wacv_embodied_source_runner import load, params, selection, hashes, GO

OFFSETS=((1,0),(-1,0),(0,1),(0,-1))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def path_check(episode, cache):
    walk,_=cache.geometry(episode["configuration"])
    position=tuple(episode["spawn"])
    path=episode["path"];actions=episode["actions"]
    assert tuple(path[0])==position and len(path)==len(actions)+1
    assert len(actions)==episode["decisions"]<=45
    reward=0
    for index,action in enumerate(actions):
        assert action in range(4)
        q=tuple(x+y for x,y in zip(position,OFFSETS[action]))
        inside=all(0<=x<n for x,n in zip(q,walk.shape))
        position=q if inside and walk[q] else position
        assert tuple(path[index+1])==position
        if position==cache.goal:
            assert index==len(actions)-1, "Acted after goal"
            reward+=1
    assert reward==episode["reward_sum"]==float(episode["success"])
    assert episode["rewards"]==[0.0]*(len(actions)-1)+[float(reward)]
    if "probabilities" in episode:
        p=np.asarray(episode["probabilities"])
        assert p.shape==(len(actions),4) and np.isfinite(p).all() and np.min(p)>=0
        assert np.max(np.abs(p.sum(-1)-1))<1e-6
        assert np.all(p[np.arange(len(actions)),actions]>0)
    if "values" in episode:
        assert len(episode["values"])==len(actions) and np.isfinite(episode["values"]).all()
    return len(actions)


def audit(out, dry=False):
    torch.set_num_threads(1)
    out=Path(out);cache,P=load();rows=[]
    expected=7 if dry else P["source"]["fits_planned"]
    binding=hashes()
    if not dry:
        go=json.loads(GO.read_text())
        assert go["hashes"]==binding and go["auditor_sha256"]==sha(__file__)
    comparison_count=0
    prediction_error=0.0
    for m in P["methods"]:
      rates=[0] if dry else range(len(m["rates"]))
      seeds=[0] if dry else P["source"]["seeds"]
      for ri in rates:
       for seed in seeds:
        d=out/m["name"]/f"rate{ri}"/f"seed{seed}"
        r=json.loads((d/"result.json").read_text())
        assert r["method"]==m["name"] and r["rate_index"]==ri and r["seed"]==seed
        assert r["rate"]==m["rates"][ri] and r["dry"]==dry and r["hashes"]==binding
        if r["status"]=="failed":
            assert not dry
            assert "error" in r and 0<=r["training_frames_completed"]<=32768
            assert sha(d/"failed_partial_episodes.json.gz")==r["partial_episodes_sha256"]
            rows.append(dict(method=m["name"],rate_index=ri,seed=seed,status="failed",
                             verified=True,selector_utility=0.0,
                             measured_accuracy=None,error=r["error"]))
            continue
        assert r["status"]=="complete"
        assert sha(d/"checkpoint_source.pt")==r["checkpoint_sha256"]
        assert sha(d/"episodes.json.gz")==r["episodes_sha256"]
        E=json.load(gzip.open(d/"episodes.json.gz","rt"))
        train=[e for e in E["train"] if "actions" in e]
        frames=1024 if dry and m["name"]=="recurrent_ppo" else 128 if dry else 32768
        assert r["train"]["frames"]==frames==sum(path_check(e,cache) for e in train)
        assert r["train"]["resources"]["counters"]["frames"]==frames
        assert all(e["configuration"] in (0,1,2,3,4,5) for e in train)
        slot_frames={}
        for e in train:
            before=slot_frames.get(e["slot"],0)
            slot_frames[e["slot"]]=before+e["decisions"]
            start_frame=before*(8 if m["name"]=="recurrent_ppo" else 1)
            if not dry:
                allowed=next(p["configurations"] for p in P["source"]["curriculum"]
                             if start_frame<p["until_total_frames"])
                assert e["configuration"] in allowed
        net=torch.load(d/"checkpoint_source.pt",map_location="cpu",weights_only=False)
        assert all(torch.isfinite(x).all() for x in params(net))
        replay=selection(net,cache,P,m["name"],ri,seed,trials=2 if dry else None)
        for k,episodes in E["selection"].items():
            actual=replay[k]["episodes"]
            assert len(episodes)==len(actual)==(2 if dry else 20)
            assert sum(path_check(e,cache) for e in episodes)==r["selection"][k]["decisions"]
            for a,b in zip(episodes,actual):
                for field in ("configuration","spawn","rotation_index","path","actions","success","decisions","rewards"):
                    assert a[field]==b[field],(m["name"],k,field)
                for field in ("probabilities","values"):
                    aa,bb=np.asarray(a[field]),np.asarray(b[field])
                    err=float(np.max(np.abs(aa-bb)))
                    assert err<=1e-7,(m["name"],k,field,err)
                    prediction_error=max(prediction_error,err);comparison_count+=aa.size
            assert abs(np.mean([x["success"] for x in episodes])-r["selection"][k]["success"])<1e-12
            assert replay[k]["counters_delta"]==r["selection"][k]["counters_delta"]
            if k.startswith("frozen"):
                assert r["selection"][k]["frozen_weights_identical"]
        rows.append(dict(method=m["name"],rate_index=ri,seed=seed,status="complete",verified=True))
        print(json.dumps({"audited":len(rows),"expected":expected,
                          "method":m["name"],"rate_index":ri,"seed":seed}),flush=True)
    assert len(rows)==expected
    rep=dict(ok=True,dry=dry,cases=len(rows),expected=expected,
             failed=sum(x["status"]=="failed" for x in rows),
             selection_prediction_elements=comparison_count,max_prediction_replay_error=prediction_error,
             rows=rows,scope="Source accounting/replay, not target or species-transfer proof")
    (out/"root_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps({k:v for k,v in rep.items() if k!="rows"}))
    return rep


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True)
    ap.add_argument("--dry",action="store_true");a=ap.parse_args()
    audit(a.out,a.dry)
