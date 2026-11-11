"""Phase 5: why does a model win on some series and lose on others?

Usage:  python -m src.analysis.why
Uses reports/tables/series_metrics_cv.csv (per series x model, CV folds pooled) + series_characteristics.csv.
Outputs: reports/tables/why_*.csv, reports/why_summary.json, figures 11-13.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from src.plotting import INK2, MODEL_COLORS, PALETTE, plt, save, setup
from src.utils import ROOT, load_config, p

TAB = "reports/tables"
FIG = "reports/figures"
FEATS = ["seasonal_strength", "trend_strength", "share_zeros", "cv", "adi", "cv2", "log_mean", "n_weeks", "abs_yoy"]


def parse_orders(info: str):
    m = re.search(r"ARIMA\((\d),(\d),(\d)\)(?:\((\d),(\d),(\d)\)\[52\])?", str(info))
    if not m:
        return None
    p_, d, q, P, D, Q = (int(x) if x is not None else 0 for x in m.groups())
    return {"p": p_, "d": d, "q": q, "P": P, "D": D, "Q": Q, "seasonal_part": (P + D + Q) > 0}


def main() -> None:
    setup()
    cfg = load_config()
    s = pd.read_csv(ROOT / f"{TAB}/series_metrics_cv.csv")
    ho = pd.read_csv(ROOT / f"{TAB}/series_metrics_holdout.csv")
    ch = pd.read_csv(ROOT / "reports/series_characteristics.csv")
    ch["log_mean"] = np.log10(ch["mean"].clip(lower=1e-3))
    ch["abs_yoy"] = ch.yoy_level_change.abs()
    W = s.pivot(index="unique_id", columns="model", values="MASE")
    Wh = ho.pivot(index="unique_id", columns="model", values="MASE")
    models = [c for c in W.columns]
    out: dict = {}

    # ------------------------------------------------ relative performance vs seasonal naive
    rel = np.log(W.div(W["SeasonalNaive"], axis=0))  # < 0 : beats seasonal naive
    d = ch.set_index("unique_id").join(rel.add_prefix("lr_")).join(W.add_prefix("mase_"))
    d["group"] = np.where(d.level == "sku_store", "SKU", "Aggregate")

    # winner per series
    d["winner"] = W.idxmin(axis=1)
    d["winner_holdout"] = Wh.idxmin(axis=1)
    fam = {"Naive": "baseline", "SeasonalNaive": "baseline", "MovingAverage": "baseline", "SES": "baseline",
           "ETS": "statistical", "SARIMA": "statistical", "DHR-ARIMA": "statistical", "Prophet": "statistical",
           "LightGBM": "ML", "Ensemble": "combination"}
    d["winner_family"] = d.winner.map(fam)

    # ------------------------------------------------ SARIMA: where does it win?
    d["sarima_vs_snaive"] = d["mase_SARIMA"] < d["mase_SeasonalNaive"]
    d["sarima_vs_lgbm"] = d["mase_SARIMA"] < d["mase_LightGBM"]
    d["sarima_vs_ets"] = d["mase_SARIMA"] < d["mase_ETS"]
    best_other = W.drop(columns=["SARIMA", "Ensemble"], errors="ignore").min(axis=1)
    d["sarima_best_or_within5pct"] = W["SARIMA"] <= 1.05 * best_other

    bins = {
        "seasonal_strength": [-0.01, 0.2, 0.4, 0.6, 1.0],
        "share_zeros": [-0.01, 0.0001, 0.2, 0.4, 1.0],
        "trend_strength": [-0.01, 0.3, 0.6, 0.8, 1.0],
        "cv": [0, 0.25, 0.6, 1.0, 10],
    }
    tabs = []
    for feat, b in bins.items():
        g = d.assign(bin=pd.cut(d[feat], b)).groupby("bin", observed=True)
        t = g.agg(n=("level", "size"), sarima_beats_snaive=("sarima_vs_snaive", "mean"), sarima_beats_lgbm=("sarima_vs_lgbm", "mean"),
                  sarima_beats_ets=("sarima_vs_ets", "mean"), median_mase_SARIMA=("mase_SARIMA", "median"),
                  median_mase_LightGBM=("mase_LightGBM", "median"), median_mase_ETS=("mase_ETS", "median"),
                  median_mase_SeasonalNaive=("mase_SeasonalNaive", "median"),
                  median_mase_DHR=("mase_DHR-ARIMA", "median"), median_mase_Prophet=("mase_Prophet", "median")).reset_index()
        t.insert(0, "feature", feat)
        t["bin"] = t["bin"].astype(str)
        tabs.append(t)
    bt = pd.concat(tabs)
    bt.to_csv(p(f"{TAB}/why_binned_win_rates.csv"), index=False)

    cls = d.assign(cls=np.where(d.level == "sku_store", "SKU-" + d.sbc_class, d.level)).groupby("cls").agg(
        n=("level", "size"), sarima_beats_snaive=("sarima_vs_snaive", "mean"), sarima_beats_lgbm=("sarima_vs_lgbm", "mean"),
        sarima_beats_ets=("sarima_vs_ets", "mean"), sarima_best_or_within5pct=("sarima_best_or_within5pct", "mean"),
        seasonal_strength=("seasonal_strength", "median"), share_zeros=("share_zeros", "median"),
        trend_strength=("trend_strength", "median"))
    cls.to_csv(p(f"{TAB}/why_by_class.csv"))
    out["by_class"] = cls.round(3).to_dict("index")

    # winners by class (CV and holdout)
    wc = pd.crosstab(np.where(d.level == "sku_store", "SKU-" + d.sbc_class, d.level), d.winner)
    wc.to_csv(p(f"{TAB}/why_winner_counts_cv.csv"))
    wch = pd.crosstab(np.where(d.level == "sku_store", "SKU-" + d.sbc_class, d.level), d.winner_holdout)
    wch.to_csv(p(f"{TAB}/why_winner_counts_holdout.csv"))
    out["winner_counts_cv"] = wc.to_dict("index")
    out["winner_stability_cv_to_holdout"] = float((d.winner == d.winner_holdout).mean())

    # ------------------------------------------------ rank correlations of relative error with characteristics
    corr = []
    for mdl in [m for m in models if m != "SeasonalNaive"]:
        for f in FEATS:
            for grp, sub in [("all", d), ("Aggregate", d[d.group == "Aggregate"]), ("SKU", d[d.group == "SKU"])]:
                x = sub[[f, f"lr_{mdl}"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(x) > 10:
                    r, pv = stats.spearmanr(x[f], x[f"lr_{mdl}"])
                    corr.append({"model": mdl, "feature": f, "group": grp, "spearman_rho": r, "p": pv, "n": len(x)})
    corr = pd.DataFrame(corr)
    corr.to_csv(p(f"{TAB}/why_spearman_relative_error.csv"), index=False)
    # SARIMA vs LightGBM head-to-head
    d["lr_sarima_lgbm"] = np.log(d["mase_SARIMA"] / d["mase_LightGBM"])
    h2h = []
    for f in FEATS:
        for grp, sub in [("all", d), ("Aggregate", d[d.group == "Aggregate"]), ("SKU", d[d.group == "SKU"])]:
            x = sub[[f, "lr_sarima_lgbm"]].dropna()
            r, pv = stats.spearmanr(x[f], x["lr_sarima_lgbm"])
            h2h.append({"feature": f, "group": grp, "spearman_rho": r, "p": pv, "n": len(x)})
    h2h = pd.DataFrame(h2h)
    h2h.to_csv(p(f"{TAB}/why_sarima_vs_lgbm_spearman.csv"), index=False)
    out["sarima_vs_lgbm_spearman"] = h2h.round(3).to_dict("records")

    # ------------------------------------------------ SARIMA orders actually selected (holdout origin + all CV)
    fc = pd.read_parquet(ROOT / cfg["paths"]["cache_dir"] / "stat_forecasts.parquet", columns=["unique_id", "origin", "model", "info"])
    orders = fc[fc.model == "SARIMA"].drop_duplicates(["unique_id", "origin"])
    od = pd.DataFrame([{**(parse_orders(i) or {}), "unique_id": u, "origin": o, "info": i} for u, o, i in
                       zip(orders.unique_id, orders.origin, orders["info"])])
    od = od.merge(ch[["unique_id", "level", "sbc_class", "seasonal_strength"]], on="unique_id")
    od.to_csv(p(f"{TAB}/sarima_selected_orders.csv"), index=False)
    od["grp"] = np.where(od.level == "sku_store", "SKU", "Aggregate")
    out["sarima_orders"] = {
        "share_with_D1": od.groupby("grp").D.mean().round(3).to_dict(),
        "share_with_any_seasonal_term": od.groupby("grp").seasonal_part.mean().round(3).to_dict(),
        "share_d1": od.groupby("grp").d.mean().round(3).to_dict(),
        "most_common_holdout": od[od.origin == "holdout"]["info"].str.replace(r"\s+", " ", regex=True).value_counts().head(8).to_dict(),
    }
    # does SARIMA help more when it actually includes a seasonal component?
    sp = od[od.origin != "holdout"].groupby("unique_id").seasonal_part.mean().rename("share_folds_seasonal")
    d = d.join(sp)
    out["sarima_seasonal_part_vs_win"] = d.assign(seas=d.share_folds_seasonal >= 0.5).groupby(["group", "seas"]).agg(
        n=("level", "size"), beats_snaive=("sarima_vs_snaive", "mean"), median_rel=("lr_SARIMA", "median")).round(3).reset_index().to_dict("records")

    # ------------------------------------------------ interpretable decision guide: shallow tree on winner family
    X = d[["seasonal_strength", "trend_strength", "share_zeros", "cv", "log_mean"]].fillna(0)
    y = d["winner"].replace({"Ensemble": "Ensemble"})
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=cfg["seed"])
    acc = cross_val_score(tree, X, y, cv=5)
    tree.fit(X, y)
    rules = export_text(tree, feature_names=list(X.columns), show_weights=True)
    open(p(f"{TAB}/why_decision_tree.txt"), "w").write(
        f"5-fold CV accuracy predicting the CV-winning model: {acc.mean():.3f} (+/- {acc.std():.3f}); "
        f"majority-class baseline: {y.value_counts(normalize=True).iloc[0]:.3f}\n\n{rules}")
    out["decision_tree_cv_accuracy"] = float(acc.mean())
    out["decision_tree_majority_baseline"] = float(y.value_counts(normalize=True).iloc[0])

    d.reset_index().to_csv(p(f"{TAB}/why_series_table.csv"), index=False)

    # ------------------------------------------------ figures
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    lv_col = {"total": PALETTE[0], "category": PALETTE[1], "department": PALETTE[2], "store": PALETTE[3],
              "store_dept": PALETTE[4], "sku_store": PALETTE[5]}
    for ax, feat, lab in zip(axes, ["seasonal_strength", "share_zeros", "trend_strength"],
                             ["Seasonal strength F_S", "Share of zero weeks", "Trend strength F_T"]):
        for lv, c in lv_col.items():
            sub = d[d.level == lv]
            ax.scatter(sub[feat], sub["lr_SARIMA"], s=14, color=c, label=lv, alpha=0.85, edgecolor="white", lw=0.3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel(lab)
        ax.set_ylabel("log(MASE SARIMA / MASE SeasonalNaive)")
        x = d[[feat, "lr_SARIMA"]].dropna()
        r = stats.spearmanr(x[feat], x["lr_SARIMA"])[0]
        ax.set_title(f"SARIMA vs seasonal naive (Spearman ρ={r:.2f})", fontsize=10)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.text(0.5, -0.02, "Below 0 = SARIMA more accurate than seasonal naive on that series (CV folds)", ha="center", color=INK2)
    save(fig, p(f"{FIG}/11_sarima_relative_error_vs_characteristics.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, feat, lab in zip(axes, ["seasonal_strength", "share_zeros"], ["Seasonal strength F_S", "Share of zero weeks"]):
        for lv, c in lv_col.items():
            sub = d[d.level == lv]
            ax.scatter(sub[feat], sub["lr_sarima_lgbm"], s=14, color=c, label=lv, alpha=0.85, edgecolor="white", lw=0.3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel(lab)
        ax.set_ylabel("log(MASE SARIMA / MASE LightGBM)")
        ax.set_title("Above 0 = LightGBM better; below 0 = SARIMA better", fontsize=10)
    axes[0].legend(fontsize=7.5)
    save(fig, p(f"{FIG}/12_sarima_vs_lightgbm.png"))

    # win-share bar chart by class
    order = ["total", "category", "department", "store", "store_dept", "SKU-smooth", "SKU-erratic", "SKU-intermittent", "SKU-lumpy"]
    wc2 = wc.reindex([o for o in order if o in wc.index])
    share = wc2.div(wc2.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    left = np.zeros(len(share))
    for mdl in [m for m in ["SeasonalNaive", "Naive", "MovingAverage", "SES", "ETS", "SARIMA", "DHR-ARIMA", "Prophet", "LightGBM", "Ensemble"] if m in share.columns]:
        ax.barh(range(len(share)), share[mdl], left=left, color=MODEL_COLORS[mdl], label=mdl, height=0.7, edgecolor="#fcfcfb", lw=1)
        left += share[mdl].to_numpy()
    ax.set_yticks(range(len(share)), [f"{i} (n={int(wc2.loc[i].sum())})" for i in share.index])
    ax.invert_yaxis()
    ax.set_xlabel("Share of series where the model has the lowest CV MASE")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8)
    ax.set_title("Which model wins, by series class (CV folds)")
    save(fig, p(f"{FIG}/13_winner_share_by_class.png"))

    json.dump(out, open(p("reports/why_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str)[:6000])
    print(bt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
