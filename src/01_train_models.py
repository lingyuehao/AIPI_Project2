# Loads the raw data, builds features and trains the models.
# Two feature sets:
#   - "booking": stuff you'd know when you buy the ticket
#   - "dayof":   booking features + weather at the origin airport at departure time
# For each one I train an EBM (glass box) and a LightGBM (black box) to compare.
import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyreadr
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

f = pyreadr.read_r("data/raw/flights.rda")["flights"]
w = pyreadr.read_r("data/raw/weather.rda")["weather"]
al = pyreadr.read_r("data/raw/airlines.rda")["airlines"]
ap = pyreadr.read_r("data/raw/airports.rda")["airports"]

for c in ["carrier", "origin", "dest"]:
    f[c] = f[c].astype(str)

# no dep_time = cancelled, departed but no arr_delay = diverted
f["cancelled"] = f.dep_time.isna()
f["diverted"] = (~f.cancelled) & f.arr_delay.isna()
f["date"] = pd.to_datetime(dict(year=f.year, month=f.month, day=f.day))
f["dow"] = f.date.dt.dayofweek  # 0 = Monday
f["sched_hour"] = (f.sched_dep_time // 100).clip(5, 23)

# rough congestion feature: how many flights are scheduled out of the same
# airport in the same hour on the same day
f["slot_load"] = f.groupby(["origin", "date", "sched_hour"]).flight.transform("size")

# weather at the origin for the scheduled hour
w["origin"] = w.origin.astype(str)
w = w.groupby(["origin", "month", "day", "hour"], as_index=False)[["temp", "wind_speed", "precip", "visib"]].mean()
f = f.merge(w.rename(columns={"hour": "sched_hour"}), on=["origin", "month", "day", "sched_hour"], how="left")
# a few hours are missing in the weather table, just fill with typical values
f["wind_speed"] = f.wind_speed.fillna(f.wind_speed.median()).clip(0, 40)
f["temp"] = f.temp.fillna(f.temp.median())
f["visib"] = f.visib.fillna(10.0)
f["precip"] = f.precip.fillna(0.0)

# keep the 40 busiest destinations, everything else goes into "Other"
top_dest = f.dest.value_counts().index[:40]
f["dest_g"] = np.where(f.dest.isin(top_dest), f.dest, "Other")

# only model flights that actually flew. target = arrived 15+ min late (BTS definition)
d = f[~f.cancelled & ~f.diverted].copy()
d["y"] = (d.arr_delay >= 15).astype(int)
print("flights used:", len(d), " late rate:", round(d.y.mean(), 4))

# split by whole days so the same day's weather can't be in train and test
rng = np.random.default_rng(7)
dates = np.array(sorted(d.date.unique()))
test_dates = set(rng.choice(dates, size=int(len(dates) * 0.2), replace=False))
d["split"] = np.where(d.date.isin(test_dates), "test", "train")

BOOK = ["carrier", "origin", "dest_g", "month", "dow", "sched_hour", "slot_load"]
DAYOF = BOOK + ["visib", "wind_speed", "precip", "temp"]
CATS = ["carrier", "origin", "dest_g"]
NOMINAL = CATS + ["month", "dow"]

tr = d[d.split == "train"]
te = d[d.split == "test"]

models = {}
for name, feats in [("booking", BOOK), ("dayof", DAYOF)]:
    # EBM, no interaction terms so every prediction is just a sum of one curve per feature
    ebm = ExplainableBoostingClassifier(
        interactions=0,
        feature_types=["nominal" if c in NOMINAL else "continuous" for c in feats],
        max_bins=64, outer_bags=4, random_state=1, n_jobs=-1,
    )
    ebm.fit(tr[feats], tr.y)
    p = ebm.predict_proba(te[feats])[:, 1]

    # lightgbm needs pandas categoricals
    Xg_tr, Xg_te = tr[feats].copy(), te[feats].copy()
    for c in CATS:
        cat_type = pd.CategoricalDtype(sorted(d[c].unique()))
        Xg_tr[c] = Xg_tr[c].astype(cat_type)
        Xg_te[c] = Xg_te[c].astype(cat_type)
    gbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    gbm.fit(Xg_tr, tr.y)
    pg = gbm.predict_proba(Xg_te)[:, 1]

    print(f"{name}: EBM auc={roc_auc_score(te.y, p):.3f} brier={brier_score_loss(te.y, p):.4f} | "
          f"GBM auc={roc_auc_score(te.y, pg):.3f} brier={brier_score_loss(te.y, pg):.4f}")
    models[name] = dict(ebm=ebm, gbm=gbm, feats=feats, p=p, pg=pg, Xg_te=Xg_te)

CAR = dict(zip(al.carrier.astype(str), al.name.astype(str)))
APN = dict(zip(ap.faa.astype(str), ap.name.astype(str)))
with open("data/state.pkl", "wb") as fh:
    pickle.dump(dict(d=d, models=models, CAR=CAR, APN=APN), fh)
print("saved data/state.pkl")
