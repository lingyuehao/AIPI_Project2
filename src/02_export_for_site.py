# Turns the trained models + test results into one JSON file the web page reads.
# What goes in:
#   - the EBM lookup tables (so the browser can compute predictions itself)
#   - summary stats for the "data" section
#   - audit numbers (calibration by group, thresholds, severe delays)
#   - EBM vs TreeSHAP comparison
#   - a sample of real test flights for the replay section
#   - a small distribution shift experiment
import json
import pickle

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score

S = pickle.load(open("data/state.pkl", "rb"))
d, M, CAR, APN = S["d"], S["models"], S["CAR"], S["APN"]
te = d[d.split == "test"].copy()
out = {}


def export_ebm(e):
    """EBM term scores -> plain dicts. index 0 / last are the missing/unknown bins, I drop them."""
    terms = []
    for i, name in enumerate(e.term_names_):
        bins, scores = e.bins_[i][0], e.term_scores_[i]
        if isinstance(bins, dict):  # categorical feature
            terms.append(dict(name=name, type="cat", map={k: float(scores[v]) for k, v in bins.items()}))
        else:  # numeric feature, bins are cut points
            terms.append(dict(name=name, type="num", cuts=[float(x) for x in bins],
                              scores=[float(x) for x in scores[1:-1]]))
    return dict(intercept=float(e.intercept_[0]), terms=terms,
                importance=dict(zip(e.term_names_, [float(x) for x in e.term_importances()])))


def predict_from_export(tex, row):
    # same logic as the javascript on the page, used to double check the export
    z = tex["intercept"]
    for t in tex["terms"]:
        v = row[t["name"]]
        if t["type"] == "cat":
            z += t["map"].get(str(v), 0)
        else:
            z += t["scores"][int(np.searchsorted(t["cuts"], v, side="right"))]
    return 1 / (1 + np.exp(-z))


for k in ["booking", "dayof"]:
    tex = export_ebm(M[k]["ebm"])
    out["ebm_" + k] = tex
    sample = te.sample(300, random_state=0)
    a = np.array([predict_from_export(tex, r) for _, r in sample.iterrows()])
    b = M[k]["ebm"].predict_proba(sample[M[k]["feats"]])[:, 1]
    print(k, "max diff between exported model and real model:", np.abs(a - b).max())
    te["p_" + k] = M[k]["p"]
    te["pg_" + k] = M[k]["pg"]

# ---- overall metrics ----
met = {}
for k in ["booking", "dayof"]:
    met[k] = dict(ebm_auc=roc_auc_score(te.y, te["p_" + k]), ebm_brier=brier_score_loss(te.y, te["p_" + k]),
                  gbm_auc=roc_auc_score(te.y, te["pg_" + k]), gbm_brier=brier_score_loss(te.y, te["pg_" + k]))
met["base_rate_brier"] = brier_score_loss(te.y, np.full(len(te), d[d.split == "train"].y.mean()))
met["n_train"] = int((d.split == "train").sum())
met["n_test"] = int(len(te))
out["metrics"] = met

# ---- lookups the page needs for default values ----
out["slot_med"] = {o: {int(h): float(v) for h, v in g.groupby("sched_hour").slot_load.median().items()}
                   for o, g in d.groupby("origin")}
out["slot_range"] = [int(d.slot_load.min()), int(d.slot_load.quantile(.99))]
out["weather_med"] = {o: {int(m): {c: float(gg[c].median()) for c in ["temp", "wind_speed", "precip", "visib"]}
                          for m, gg in g.groupby("month")} for o, g in d.groupby("origin")}
out["carriers"] = {c: CAR.get(c, c) for c in sorted(d.carrier.unique())}
dg = d.groupby("dest_g").agg(n=("y", "size"), rate=("y", "mean"), dist=("distance", "median")).reset_index()
out["dests"] = {r.dest_g: dict(name=APN.get(r.dest_g, "All other destinations (78 smaller airports)"),
                               n=int(r.n), rate=float(r.rate), dist=float(r.dist)) for r in dg.itertuples()}
out["origins"] = {o: APN.get(o, o) for o in ["EWR", "JFK", "LGA"]}
# which airline actually flies which route (used to warn about made-up routes)
out["routes"] = {c: {o: sorted(g2.dest_g.unique().tolist()) for o, g2 in g.groupby("origin")}
                 for c, g in d.groupby("carrier")}


# ---- descriptive stats for the data section ----
def rate_by(col):
    g = d.groupby(col).agg(rate=("y", "mean"), n=("y", "size"), mean_delay=("arr_delay", "mean"))
    return {str(i): dict(rate=float(r.rate), n=int(r.n), mean_delay=float(r.mean_delay)) for i, r in g.iterrows()}


eda = {c: rate_by(c) for c in ["sched_hour", "month", "dow", "carrier", "origin"]}
hm = d.groupby(["month", "sched_hour"]).y.mean().unstack()
eda["heat"] = {"months": list(map(int, hm.index)), "hours": list(map(int, hm.columns)),
               "z": [[None if pd.isna(x) else round(float(x), 4) for x in row] for row in hm.values]}
edges = list(range(-60, 181, 10))
eda["hist"] = {"edges": edges, "counts": np.histogram(d.arr_delay.clip(-60, 180), bins=edges + [1e9])[0].tolist()}
eda["totals"] = dict(scheduled=435352, cancelled=10738, diverted=1796, completed=int(len(d)),
                     delayed=int(d.y.sum()), rate=float(d.y.mean()),
                     delay_hours=float(d.arr_delay.clip(lower=0).sum() / 60), median_delay=float(d.arr_delay.median()),
                     share_60=float((d.arr_delay >= 60).mean()), share_180=float((d.arr_delay >= 180).mean()))
vis_bins = pd.cut(d.visib, [-1, 1, 3, 6, 9.9, 11], labels=["<1 mi", "1–3 mi", "3–6 mi", "6–<10 mi", "10 mi (clear)"])
eda["visib"] = {str(k): dict(rate=float(v.mean()), n=int(len(v))) for k, v in d.groupby(vis_bins, observed=True).y}
pre_bins = pd.cut(d.precip, [-1, 0, 0.02, 0.1, 10], labels=["none", "trace–0.02 in", "0.02–0.1 in", ">0.1 in"])
eda["precip"] = {str(k): dict(rate=float(v.mean()), n=int(len(v))) for k, v in d.groupby(pre_bins, observed=True).y}
out["eda"] = eda


# ---- audit ----
def calib(y, p, nb=10):
    q = np.clip((p * nb).astype(int), 0, nb - 1)
    return [[float(p[q == i].mean()), float(y[q == i].mean()), int((q == i).sum())]
            for i in range(nb) if (q == i).sum() >= 30]


def ece(y, p, nb=10):
    q = np.clip((p * nb).astype(int), 0, nb - 1)
    return float(sum(abs(p[q == i].mean() - y[q == i].mean()) * (q == i).sum() for i in range(nb) if (q == i).sum()) / len(y))


dest_counts = d.dest.value_counts()
te["dest_size"] = te.dest.map(lambda x: "Busy (≥10k flights/yr)" if dest_counts[x] >= 10000
                              else ("Medium (2k–10k)" if dest_counts[x] >= 2000 else "Small (<2k)"))
te["tod"] = pd.cut(te.sched_hour, [0, 11, 16, 24],
                   labels=["Morning (5–11)", "Afternoon (12–16)", "Evening (17–23)"]).astype(str)
te["season"] = te.month.map(lambda m: "Summer (Jun–Aug)" if m in (6, 7, 8) else ("Winter (Dec–Feb)" if m in (12, 1, 2) else "Spring/Fall"))
te["carrier_name"] = te.carrier.map(lambda c: CAR.get(c, c))
te["wx"] = np.where((te.visib < 3) | (te.precip > 0.02), "Bad weather at origin", "Fair weather")

audit = {}
for k in ["booking", "dayof"]:
    p, y = te["p_" + k].values, te.y.values
    A = {"overall": {"All test flights": dict(n=len(y), actual=float(y.mean()), pred=float(p.mean()),
                                              auc=float(roc_auc_score(y, p)), ece=ece(y, p), calib=calib(y, p))}}
    for g in ["carrier_name", "origin", "dest_size", "tod", "season", "wx"]:
        A[g] = {}
        for lv, gg in te.groupby(g):
            if len(gg) < 300:  # skip tiny groups
                continue
            yy, pp = gg.y.values, gg["p_" + k].values
            A[g][lv] = dict(n=int(len(gg)), actual=float(yy.mean()), pred=float(pp.mean()),
                            auc=float(roc_auc_score(yy, pp)), ece=ece(yy, pp), calib=calib(yy, pp))
    # what did the model predict for flights that ended up really late?
    band = pd.cut(te.arr_delay, [-1e9, 0, 15, 60, 180, 1e9],
                  labels=["early/on time", "0–14 min late", "15–59 min", "1–3 h", "3 h+"])
    A["by_outcome"] = {str(b): dict(n=int(len(g)), mean_pred=float(g["p_" + k].mean()),
                                    q=[float(x) for x in g["p_" + k].quantile([.1, .25, .5, .75, .9])])
                       for b, g in te.groupby(band, observed=True)}
    # if an app showed "likely delayed" above a cutoff
    A["thresholds"] = []
    for t in [0.2, 0.3, 0.4, 0.5]:
        flag = p >= t
        A["thresholds"].append(dict(t=t, flagged=float(flag.mean()), precision=float(y[flag].mean()),
                                    recall=float(flag[y == 1].mean())))
    audit[k] = A
out["audit"] = audit

# ---- EBM vs LightGBM+TreeSHAP on the same flights ----
feats, gbm, ebm = M["dayof"]["feats"], M["dayof"]["gbm"], M["dayof"]["ebm"]
sub = te.sample(20000, random_state=3)
shap_vals = gbm.predict(M["dayof"]["Xg_te"].loc[sub.index], pred_contrib=True)[:, :-1]  # last col is bias
local = ebm.explain_local(sub[feats])
ebm_vals = np.array([local.data(i)["scores"] for i in range(len(sub))])
rho = [spearmanr(np.abs(shap_vals[i]), np.abs(ebm_vals[i])).correlation for i in range(2000)]
out["disagree"] = dict(
    top1=float((np.abs(shap_vals).argmax(1) == np.abs(ebm_vals).argmax(1)).mean()),
    spearman_median=float(np.nanmedian(rho)),
    sign=float(np.mean(np.sign(shap_vals) == np.sign(ebm_vals))),
    global_gbm={n: float(v) for n, v in zip(feats, np.abs(shap_vals).mean(0))},
    global_ebm={n: float(v) for n, v in zip(feats, np.abs(ebm_vals).mean(0))},
)
print("explainers agree on top factor:", round(out["disagree"]["top1"], 3))

# ---- sample of real flights for the replay section ----
# I oversample late flights on purpose so there are interesting cases to look at
sev = sub[sub.arr_delay >= 60].sample(90, random_state=1)
ok = sub[sub.arr_delay < 15].sample(160, random_state=1)
mid = sub[(sub.arr_delay >= 15) & (sub.arr_delay < 60)].sample(70, random_state=1)
samp = pd.concat([sev, ok, mid]).sample(frac=1, random_state=2)
pos = {ix: i for i, ix in enumerate(sub.index)}
out["flights"] = [dict(
    date=r.date.strftime("%Y-%m-%d"), flight=f"{r.carrier}{int(r.flight)}", carrier=r.carrier, origin=r.origin,
    dest=r.dest, dest_g=r.dest_g, month=int(r.month), dow=int(r.dow), sched_hour=int(r.sched_hour),
    sched_dep=int(r.sched_dep_time), slot_load=int(r.slot_load), visib=round(float(r.visib), 2),
    wind_speed=round(float(r.wind_speed), 1), precip=round(float(r.precip), 3), temp=round(float(r.temp), 1),
    arr_delay=int(r.arr_delay), dep_delay=int(r.dep_delay), gbm_p=round(float(r.pg_dayof), 4),
    gbm_shap=[round(float(x), 4) for x in shap_vals[pos[ix]]]) for ix, r in samp.iterrows()]

# ---- distribution shift test ----
# train on Jan-Jun only, no month feature (it would never see Jul-Dec values anyway), test on Jul-Dec
fb = [c for c in M["booking"]["feats"] if c != "month"]
trn, tst = d[d.month <= 6], d[d.month >= 7]
e2 = ExplainableBoostingClassifier(interactions=0, feature_types=["nominal" if c in ("carrier", "origin", "dest_g", "dow") else "continuous" for c in fb],
                                   max_bins=64, outer_bags=2, random_state=1, n_jobs=-1)
e2.fit(trn[fb], trn.y)
p_in, p_out = e2.predict_proba(trn[fb])[:, 1], e2.predict_proba(tst[fb])[:, 1]
mm = pd.DataFrame({"month": np.r_[trn.month, tst.month], "p": np.r_[p_in, p_out], "y": np.r_[trn.y, tst.y]}).groupby("month").mean()
out["shift"] = dict(months=[int(m) for m in mm.index], pred=[float(x) for x in mm.p], actual=[float(x) for x in mm.y],
                    auc_in=float(roc_auc_score(trn.y, p_in)), auc_out=float(roc_auc_score(tst.y, p_out)))

json.dump(out, open("site_data/data.json", "w"), separators=(",", ":"))
print("wrote site_data/data.json")
