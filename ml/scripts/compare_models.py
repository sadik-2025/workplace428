"""
Compares Baseline CNN (Phase 2), Augmented CNN (Phase 3), and the SSL +
Domain-Adversarial CNN (Phase 4) side by side using each model's saved
metrics.json. Produces a summary table (CSV) and a bar chart — useful
directly for your project report, and this is the same data your backend's
/models/compare endpoint will serve to the Model Comparison dashboard later.

Usage:
    python compare_models.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_DIRS = {
    "Baseline CNN": "../models/baseline/metrics.json",
    "Augmented CNN": "../models/augmented/metrics.json",
    "SSL + Domain-Adversarial CNN": "../models/ssl_domain_gen/metrics.json",
}


def main():
    rows = []
    for name, path in MODEL_DIRS.items():
        p = Path(path)
        if not p.exists():
            print(f"Skipping {name} — {p} not found yet.")
            continue
        with open(p) as f:
            m = json.load(f)
        rows.append(
            {
                "Model": name,
                "Test Accuracy": m.get("test_accuracy"),
                "Precision (macro)": m.get("precision_macro"),
                "Recall (macro)": m.get("recall_macro"),
                "F1 (macro)": m.get("f1_macro"),
                "Unseen-Domain Accuracy": m.get("unseen_domain_accuracy"),
                "Generalization Gap": m.get("generalization_gap"),
                "Inference (ms/batch)": m.get("inference_time_ms_per_batch"),
                "Parameters": m.get("num_parameters"),
                "Model Size (MB)": m.get("model_size_mb"),
            }
        )

    if not rows:
        print("No metrics.json files found yet — train the models first.")
        return

    df = pd.DataFrame(rows)
    out_dir = Path("../../docs/model_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison.csv", index=False)
    print(df.to_string(index=False))

    # Accuracy comparison chart (standard vs unseen-domain, side by side per model)
    has_unseen = "Unseen-Domain Accuracy" in df.columns and df["Unseen-Domain Accuracy"].notna().any()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    width = 0.35 if has_unseen else 0.6

    ax.bar([i - width / 2 for i in x] if has_unseen else x, df["Test Accuracy"], width, label="Standard test")
    if has_unseen:
        ax.bar([i + width / 2 for i in x], df["Unseen-Domain Accuracy"], width, label="Unseen domain")

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison: Standard vs Unseen-Domain Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_chart.png", dpi=150)
    plt.close()

    print(f"\nSaved comparison.csv and comparison_chart.png to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
