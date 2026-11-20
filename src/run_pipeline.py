"""Run the full CreditRisk-IFRS9 pipeline end-to-end on synthetic data.

Order matters (and is the point of the pipeline):
1. Generate synthetic loans + monthly snapshots.
2. Train the PD scorecard (out-of-time split) and write metrics/deciles/plots.
3. Compute expected loss (EL = PD x LGD x EAD) and macro stress scenarios.
4. Monitor population stability (PSI) month by month.

Usage:
    python -m src.run_pipeline [--n-loans 20000] [--skip-generate]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src import expected_loss as el_mod
from src import generate_synthetic as gen
from src import monitoring as mon
from src import train_pd as train
from src.utils import DATA_DIR, OUTPUTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-loans", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-generate", action="store_true",
                        help="Reuse data/synthetic/*.csv instead of regenerating")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "synthetic")
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.outputs.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        print("== 1/4 Generating synthetic data ==")
        loans = gen.generate_loans(args.n_loans, args.seed)
        snapshots = gen.generate_snapshots(loans, args.seed)
        loans.to_csv(args.data_dir / "loans.csv", index=False)
        snapshots.to_csv(args.data_dir / "snapshots.csv", index=False)
        print(f"   {len(loans):,} loans, {len(snapshots):,} snapshots, "
              f"default rate {loans['default_12m'].mean():.2%}")
    else:
        loans = train.load_loans(args.data_dir / "loans.csv")

    print("== 2/4 Training PD scorecard ==")
    metrics = train.train_and_evaluate(loans, args.outputs)

    print("== 3/4 Expected loss & stress ==")
    el_result = el_mod.run(args.data_dir / "loans.csv",
                           args.outputs / "test_predictions.csv",
                           args.data_dir / "snapshots.csv", args.outputs)

    print("== 4/4 Monitoring (PSI) ==")
    mon_result = mon.run(args.outputs / "test_predictions.csv", args.outputs)

    print("\nPipeline complete. Outputs in", args.outputs)
    print("Champion model:", metrics["champion"],
          "| ROC-AUC:", round(metrics["models"]["logistic_woe"]["roc_auc"], 3),
          "| KS:", round(metrics["models"]["logistic_woe"]["ks"], 3))
    print("Total EL (base):",
          f"${el_result['total_el']:,.0f} on EAD ${el_result['total_ead']:,.0f}")
    print("Max monthly PSI:", round(mon_result["max_psi"], 3))


if __name__ == "__main__":
    main()
