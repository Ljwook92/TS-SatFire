"""Compare the TS6 epoch-10 and epoch-21 full-crop masks on active test frames."""
import argparse
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

MODEL = "local_extent"
KEY = ["event", "target"]
FEATURES = ["distance_km", "day_delta_i4_K", "nearby_frp_MW"]
GROUPS = ["gained_tp", "added_fp", "removed_fp", "lost_tp"]
BINS = ["0-0.5", "0.5-1", "1-2", "2-4", ">4"]


def read_metrics(folder):
    f = pd.read_csv(folder / "frame_metrics.csv", dtype={"event": str, "target": str})
    f = f.loc[f.model.eq(MODEL) & f.ts.eq(6)].set_index(KEY).sort_index()
    assert len(f) == 710 and f.index.is_unique and f.index.get_level_values(0).nunique() == 24
    return f[["tp", "fp", "fn"]]


def is_run(folder, epochs, selected):
    return (folder / "frame_metrics.csv").is_file() and all((folder / name).is_file()
            for name in ("validation.json", "choices.json")) and len(json.loads(
            (folder / "validation.json").read_text())) == epochs and json.loads(
            (folder / "choices.json").read_text())["epoch"] == selected


def grid(bits):
    return np.unpackbits(bits).reshape(256, 256).astype(bool)


def read_scene(old, new, cache, event, target, rec):
    def pred(folder):
        with np.load(folder / "masks" / event / (target + ".npz")) as z:
            return grid(z[MODEL])

    idx = int(rec.target_index)
    with np.load(cache / event / f"{idx:03d}.npz") as z:
        assert str(z["date"]) == target
        truth = grid(z["new"])
    with np.load(cache / event / f"{idx-1:03d}.npz") as z:
        assert str(z["date"]) == rec.last_input
        past, local = grid(z["past"]), z["local"].astype("float32")
    a, b = pred(old), pred(new)
    assert not ((truth | a | b) & past).any()
    return truth, past, a, b, local


def changes(truth, a, b):
    return dict(zip(GROUPS, (truth & ~a & b, ~truth & ~a & b,
                             ~truth & a & ~b, truth & a & ~b)))


def pixels(masks, local, event, date):
    rows = []
    for group, mask in masks.items():
        r, c = np.where(mask)
        if not len(r):
            continue
        x = local[:, r, c]
        rows.append(pd.DataFrame(dict(event=event, target=date, row=r, col=c, group=group,
            distance_km=np.where(x[17] > 0, np.expm1(x[0]), np.nan),
            day_delta_i4_K=np.where(x[4] > 0, x[3]*12, np.nan),
            hotspot_10km=x[16].astype(int),
            nearby_frp_MW=np.where(x[16] > 0, np.expm1(x[12]*4), np.nan))))
    return rows


def draw(scene, event, date, prev, out):
    truth, past, a, b, _ = scene
    colors = ["#ffffff", "#d0d3d4", "#0072b2", "#d55e00", "#009e73", "#7055a1", "#eabf6a"]
    # History, truth, and predictions occupy the same extent in every panel.
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.8), constrained_layout=True)
    r, c = np.where(truth | a | b)
    view = (max(0,c.min()-6)-.5, min(255,c.max()+6)+.5,
            min(255,r.max()+6)+.5, max(0,r.min()-6)-.5)
    for ax, pred, title in zip(axs, (a,b,None), ("Epoch 10", "Epoch 21", "Changed predictions")):
        img = np.zeros(past.shape, dtype=int)
        img[past] = 1
        img[truth] = 2
        if pred is None:
            ch = changes(truth,a,b)
            for group, color in zip(GROUPS, (4,3,6,5)):
                img[ch[group]] = color
        else:
            img[pred & ~truth] = 3
            img[pred & truth] = 4
        ax.imshow(img, cmap=ListedColormap(colors), interpolation="nearest", vmin=0, vmax=6)
        ax.axis(view)
        ax.set_title(title)
        ax.set_xlabel("Crop column")
    axs[0].set_ylabel("Crop row")
    def legend(ax, items):
        ax.legend(handles=[Patch(color=colors[k], label=label) for k,label in items],
                  loc="upper center", bbox_to_anchor=(.5,-.15), fontsize=8, ncol=2)
    legend(axs[0], [(1,"Prior fire"),(2,"Missed spread"),(4,"Hit"),(3,"False positive")])
    legend(axs[2], [(1,"Prior fire"),(2,"Other actual spread"),(4,"Gained hit"),
                    (5,"Lost hit"),(3,"Added false positive"),(6,"Removed false positive")])
    fig.suptitle(f"{event}   |   input through {prev}   |   target {date}")
    fig.savefig(out / f"map_{event}_{date}.png", dpi=190)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True, help="100-epoch run root")
    ap.add_argument("--old", type=Path, help="10-epoch run root (optional)")
    args = ap.parse_args()
    prep, new = args.prep, args.new / "TS6"
    options = [args.old/"TS6"] if args.old else [p.parent for root in
               (args.new.parent, prep.parent) for p in root.glob("*/TS6/choices.json")]
    matches = set()
    for p in options:
        if is_run(p, 10, 10):
            m = read_metrics(p)
            if (m.fn.sum(), m.fp.sum()) == (26142, 41130):
                matches.add(p.resolve())
    if len(matches) != 1:
        raise SystemExit(f"Specify --old /path/to/10-epoch-run. Matches: {matches}")
    old, = matches
    assert is_run(new, 100, 21)
    a, b = read_metrics(old), read_metrics(new)
    pd.testing.assert_index_equal(a.index, b.index)
    assert (a.tp+a.fn).equals(b.tp+b.fn) and (b.fn.sum(),b.fp.sum()) == (24638,46436)
    active = a.index[(a.tp+a.fn) > 0]
    assert len(active) == 534 and active.get_level_values(0).nunique() == 21
    manifest = pd.read_csv(prep/"code/assets/windows.csv.gz", dtype={"event":str,"target_date":str})
    w = manifest.loc[manifest.ts.eq(6) & manifest.split.eq("test")].set_index(["event","target_date"])
    assert w.index.is_unique and set(w.index) == set(a.index)
    print("10-epoch:", old, "\nEpoch 21:", new, flush=True)
    frames, samples = [], []
    for event, date in active:
        scene = read_scene(old, new, prep/"cache", event, date, w.loc[(event,date)])
        truth, _, pa, pb, local = scene
        for pred, table in ((pa,a), (pb,b)):
            check = (int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum()))
            assert check == tuple(table.loc[(event,date)]), (event,date,check)
        ch = changes(truth,pa,pb)
        frames.append(dict(event=event, target=date, truth=int(truth.sum()),
                           **{k:int(v.sum()) for k,v in ch.items()}))
        samples.extend(pixels(ch,local,event,date))
    f, p = pd.DataFrame(frames), pd.concat(samples, ignore_index=True)
    f["net_fp"] = f.added_fp - f.removed_fp
    f["net_tp"] = f.gained_tp - f.lost_tp
    assert (f.net_fp.sum(), f.gained_tp.sum(), f.lost_tp.sum()) == (5201,2246,742)
    out = Path(tempfile.mkdtemp(prefix="fp_locations_", dir=args.new)).resolve()
    print("Output:", out)
    f.to_csv(out/"by_date.csv", index=False)
    events = f.groupby("event")[["truth",*GROUPS,"net_tp","net_fp"]].sum().sort_values("net_fp",ascending=False)
    events.to_csv(out/"by_event.csv")
    print("\nAll active frames:\n",events.sum().to_string(), "\nTop events by net FP:\n",events.head().to_string())
    p["ring_km"] = pd.cut(p.distance_km,[0,.5,1,2,4,np.inf], labels=BINS, include_lowest=True)
    p.to_csv(out/"changed_pixels.csv.gz",index=False)
    ring = p.assign(ring=p.ring_km.astype(object).fillna("no_prior_fire")).groupby(
        ["ring","group"]).size().unstack(fill_value=0).reindex(columns=GROUPS,fill_value=0)
    ring["net_fp"] = ring.added_fp-ring.removed_fp
    ring = ring.reindex([s for s in BINS+["no_prior_fire"] if s in ring.index])
    ring.to_csv(out/"by_distance.csv")
    print("\nApproximate distance bands (km):\n",ring.to_string())
    q = p[p.group.isin(["gained_tp","added_fp"])]
    rows = []
    for group, sub in q.groupby("group"):
        for feat in FEATURES + ["hotspot_10km"]:
            x = sub[feat].dropna()
            rows.append(dict(group=group,feature=feat,n=len(x),total=len(sub),mean=x.mean(),
                             q25=x.quantile(.25),median=x.median(),q75=x.quantile(.75)))
    summary = pd.DataFrame(rows)
    summary.to_csv(out/"feature_summary.csv",index=False)
    print("\nPixel summary:\n",summary.round(3).to_string(index=False))
    paired = []
    for keys, fs, mode in [(KEY, FEATURES+["hotspot_10km"], "same_day"),
                            (KEY+["ring_km"], FEATURES[1:]+["hotspot_10km"], "same_day_ring")]:
        for feat in fs:
            # For the binary indicator compare detection fractions, not medians.
            g = q.groupby(keys+["group"], observed=True)[feat]
            stat = g.mean() if feat=="hotspot_10km" else g.median()
            pair = stat.unstack("group").reindex(columns=["gained_tp","added_fp"]).dropna()
            pair["difference"] = pair.gained_tp - pair.added_fp
            pair["feature"], pair["comparison"] = feat, mode
            paired.append(pair.reset_index())
    pairs = pd.concat(paired,ignore_index=True)
    pairs.to_csv(out/"paired_features.csv",index=False)
    print("\nGained TP minus added FP (median paired difference; hotspot is a fraction):\n",
          pairs.groupby(["comparison","feature"]).difference.agg(["count","median"]).round(3).to_string())
    # One high-net-FP case and one contrasting event with many recovered hits.
    first = events.index[0]
    second = events.drop(index=first).sort_values("gained_tp",ascending=False).index[0]
    for event, sortcol in [(first,"net_fp"), (second,"gained_tp")]:
        best = f[f.event.eq(event)].sort_values([sortcol,"target"],ascending=[False,True]).iloc[0]
        date = best.target
        rec = w.loc[(event,date)]
        draw(read_scene(old,new,prep/"cache",event,date,rec), event,date,rec.last_input,out)
        print("Map:",event,date, "selected by",sortcol, "| gained TP",best.gained_tp,
              "| added FP",best.added_fp,"| removed FP",best.removed_fp)
    print("Results:", out)


if __name__ == "__main__":
    main()
