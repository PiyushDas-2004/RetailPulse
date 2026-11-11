"""Ablation: LightGBM without the price features (price_rel52, price_chg).

Why: those two features use the shelf price of the TARGET week, which is only legitimate if the retailer knows its
own price plan for the next quarter (an assumption). If accuracy barely changes without them, conclusions about
LightGBM do not hinge on that assumption.
Usage:  python -m src.models.lgbm_ablation      Output: reports/tables/lgbm_ablation_no_price.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.load import load_calendar, load_prices, load_series, make_origins
from src.evaluation.metrics import mase_scale
from src.features.lgbm_features import CAT_FEATURES, Panel, feature_names, origin_features, training_frame
from src.models.lightgbm_global import make_model
from src.utils import ROOT, load_config, p

DROP = ["price_rel52", "price_chg"]


def main() -> None:
    cfg = load_config()
    h = cfg["problem"]["horizon"]
    df, meta = load_series()
    pn = Panel(df, meta, load_calendar(), load_prices(), cfg["data"]["n_full_weeks"])
    params = json.load(open(ROOT / "reports/tables/lgbm_best_params.json"))
    rows = []
    for o in [o for o in make_origins(cfg) if o.kind != "tuning"]:
        tr = training_frame(pn, o.train_end, range(1, h + 1))
        te = origin_features(pn, o.train_end, range(1, h + 1))
        cap = np.quantile(tr.y_scaled, 0.999)
        for variant, feats in [("with_price", feature_names(tr)), ("no_price", [f for f in feature_names(tr) if f not in DROP])]:
            m = make_model(params, cfg["seed"])
            m.fit(tr[feats], np.minimum(tr.y_scaled, cap), categorical_feature=CAT_FEATURES)
            pred = np.maximum(m.predict(te[feats]) * te.scale.to_numpy(), 0)
            y = pn.Y[te.series.to_numpy(), te.target_week.to_numpy()]
            d = pd.DataFrame({"series": te.series, "y": y, "f": pred})
            for s, g in d.groupby("series"):
                ytr = pn.Y[s, : o.train_end + 1]
                ytr = ytr[~np.isnan(ytr)]
                rows.append({"origin": o.name, "variant": variant, "unique_id": pn.ids[s],
                             "MASE": np.mean(np.abs(g.y - g.f)) / mase_scale(ytr, 52)})
        print(o.name, "done")
    r = pd.DataFrame(rows)
    r["group"] = np.where(r.unique_id.str.startswith("SKU_"), "SKU", "Aggregate")
    r["period"] = np.where(r.origin == "holdout", "holdout", "cv")
    out = r.groupby(["period", "group", "variant"]).MASE.mean().unstack().round(4)
    out.to_csv(p("reports/tables/lgbm_ablation_no_price.csv"))
    print(out)


if __name__ == "__main__":
    main()
