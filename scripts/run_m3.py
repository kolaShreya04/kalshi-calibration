"""
M3 entry point: residual prediction model.

Prereq: data/processed/early_prices.parquet

Outputs:
  data/processed/m3_fold_metrics.csv     per-fold, per-model metrics
  data/processed/m3_model_summary.csv    mean±sd + gain vs price-only baseline
  data/processed/m3_importances.csv      permutation importances
  figures/m3_importances.png
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import early_price, residual_model  # noqa: E402

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(f"{config.PROCESSED_DIR}/early_prices.parquet")
    df = df[early_price.usable_mask(df)]
    log.info("sample after usable-market filter: %d markets", len(df))

    price_col = config.PRIMARY_PRICE_COL
    folds, imps = residual_model.evaluate(df, price_col,
                                          n_splits=args.n_splits)
    summary = residual_model.summarize_folds(folds)

    out = Path(config.PROCESSED_DIR)
    folds.to_csv(out / "m3_fold_metrics.csv", index=False)
    summary.to_csv(out / "m3_model_summary.csv", index=False)
    if not imps.empty:
        imps.to_csv(out / "m3_importances.csv", index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from src.viz import STYLE, _save
            plt.rcParams.update(STYLE)
            fig, ax = plt.subplots()
            d = imps.sort_values("importance")
            ax.barh(d["feature"], d["importance"], xerr=d["importance_sd"])
            ax.set_xlabel("Permutation importance (Δ neg-log-loss)")
            ax.set_title("What drives predictable mispricing? (grad_boosting)")
            _save(fig, "m3_importances")
        except Exception as e:  # noqa: BLE001
            log.warning("importance plot failed: %s", e)

    cols = ["model", "auc_mean", "auc_sd", "log_loss_mean",
            "auc_gain_vs_price_only", "ll_gain_vs_price_only"]
    print("\n=== M3 model comparison (temporal CV) ===")
    print(summary[cols].to_string(index=False))
    print("\nInterpretation: positive gains vs price_only_logistic on held-out"
          "\nfolds = systematic, feature-attributable mispricing at open.")


if __name__ == "__main__":
    main()
