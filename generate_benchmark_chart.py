"""
Generate verified training & evaluation charts for the Lexsi AI Safety Hackathon.

All values are recorded directly from executed Google Colab runs:
- Panel A: AlignTune DPO v2 training logs (160 steps, 2 epochs on 80 pairs from
  Eterniti/curated-cohere-safety-dpo).
- Panel B: SafeTune inference-time steering sweep across Layer 14 and Layers [13,14,15]
  on the 6 hard adversarial failure probes + 5 benign boundary probes, annotated
  with AuditKIT RedTeamRunner (39/47 = 83.0%) and SafeTune Layer-14 ablation (0/5).
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def build_chart(output_path: str = "benchmark_results.png") -> None:
    """Render two-panel training and SafeTune Pareto chart from verified metrics."""
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4), dpi=300)

    # --- Panel A: AlignTune DPO v2 Training Convergence (160 steps, 2 epochs) ---
    steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    losses = [
        0.6940, 0.6840, 0.6881, 0.6840, 0.6735, 0.6715, 0.6788, 0.6733,
        0.6612, 0.6595, 0.6535, 0.6733, 0.6667, 0.6508, 0.6392, 0.6539,
    ]
    accuracies = [
        40.0, 70.0, 50.0, 70.0, 70.0, 80.0, 90.0, 80.0,
        90.0, 100.0, 100.0, 90.0, 100.0, 100.0, 90.0, 100.0,
    ]
    margins = [
        -0.0009, 0.0188, 0.0106, 0.0187, 0.0402, 0.0439, 0.0294, 0.0408,
        0.0654, 0.0685, 0.0816, 0.0404, 0.0540, 0.0869, 0.1123, 0.0806,
    ]

    c_acc = "#2563eb"
    c_loss = "#dc2626"

    l1 = ax1.plot(
        steps, accuracies, color=c_acc, marker="o", linewidth=2.2,
        markersize=5, label="Reward Accuracy (%)"
    )
    ax1.set_title(
        "AlignTune DPO v2 Convergence (80 Pairs, 2 Epochs)",
        fontsize=12.5, fontweight="bold", pad=12,
    )
    ax1.set_xlabel("Training Step (Epoch 1 = Step 80, Epoch 2 = Step 160)", fontsize=10.2, fontweight="bold")
    ax1.set_ylabel("DPO Reward Accuracy (%)", fontsize=10.2, fontweight="bold", color=c_acc)
    ax1.tick_params(axis="y", labelcolor=c_acc)
    ax1.set_ylim(30, 112)
    ax1.set_xticks(steps[::2])
    ax1.axvline(80, color="#64748b", linestyle=":", linewidth=1.2, alpha=0.85)
    ax1.text(82, 34, "Epoch 2", fontsize=8.5, color="#475569")

    ax1_r = ax1.twinx()
    l2 = ax1_r.plot(
        steps, losses, color=c_loss, marker="s", linestyle="--",
        linewidth=1.8, markersize=4.5, label="DPO Loss"
    )
    ax1_r.set_ylabel("DPO Training Loss", fontsize=10.2, fontweight="bold", color=c_loss)
    ax1_r.tick_params(axis="y", labelcolor=c_loss)
    ax1_r.set_ylim(0.62, 0.71)
    ax1_r.grid(False)

    ax1.text(
        96, 46.0,
        f"Peak Margin: +{max(margins):.4f} (Step 150)\nMin Loss: {min(losses):.4f} | Final Acc: {accuracies[-1]:.0f}%",
        fontsize=8.3, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.38", fc="#f8fafc", ec="#cbd5e1"),
    )

    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", frameon=True, facecolor="white", fontsize=9)

    # --- Panel B: SafeTune Inference-Time Steering Pareto Tradeoff ---
    configs = [
        "DPO v2\nBaseline\n(s=0.0)",
        "Layer 14\nSteer\n(s=3.0)",
        "Layer 14\nSteer\n(s=6.0)",
        "Layers 13–15\nSteer*\n(s=2.0)",
        "Layers 13–15\nSteer\n(s=3.5)",
    ]
    hard_adv_pct = [100 * 1 / 6, 100 * 2 / 6, 100 * 2 / 6, 100 * 3 / 6, 100 * 5 / 6]
    benign_pct = [100 * 5 / 5, 100 * 5 / 5, 100 * 4 / 5, 100 * 5 / 5, 100 * 3 / 5]

    x = np.arange(len(configs))
    width = 0.35

    r_adv = ax2.bar(
        x - width / 2, hard_adv_pct, width,
        label="Hard Adversarial Blocked (n=6)",
        color="#10b981", edgecolor="#064e3b", linewidth=0.75,
    )
    r_ben = ax2.bar(
        x + width / 2, benign_pct, width,
        label="Benign Utility Retained (n=5)",
        color="#3b82f6", edgecolor="#1e3a8a", linewidth=0.75,
    )

    # Highlight Pareto sweet spot
    ax2.axvspan(2.55, 3.45, color="#fef3c7", alpha=0.55, zorder=0)
    ax2.text(3.0, 108.5, "Pareto Sweet Spot\n(2/2 Persona Blocked)", ha="center", va="bottom",
             fontsize=8.2, fontweight="bold", color="#92400e")

    ax2.set_title(
        "SafeTune Layer-Targeted Steering Sweep (DPO v2 + Steer)",
        fontsize=12.5, fontweight="bold", pad=12,
    )
    ax2.set_ylabel("Score (%) — Higher is Better", fontsize=10.2, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=8.8)
    ax2.set_ylim(0, 124)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for rect, raw in zip(list(r_adv) + list(r_ben), ["1/6", "2/6", "2/6", "3/6", "5/6", "5/5", "5/5", "4/5", "5/5", "3/5"]):
        h = rect.get_height()
        ax2.annotate(
            f"{h:.1f}%\n({raw})",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 2), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.3, fontweight="bold",
        )

    fig.suptitle(
        "Calibrated Safety Alignment Audit: CohereLabs/tiny-aya-earth (3.35B)\n"
        "Stack: CuratorKIT (80 pairs) -> AlignTune (DPO v2) -> SafeTune (Layer 13–15) -> AuditKIT",
        fontsize=13.5, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.015,
        "Verified Baseline Invariants: Direct Safety Refusal = 100.0% (20/20 across Cyber, CBRN, Clinical, Multilingual)  |  "
        "AuditKIT RedTeam = 83.0% (39/47)  |  SafeTune Layer-14 Ablation Cyber Refusal = 0.0% (0/5)",
        ha="center", va="bottom", fontsize=8.4, color="#334155",
        bbox=dict(boxstyle="round,pad=0.32", fc="#f8fafc", ec="#cbd5e1"),
    )
    plt.tight_layout(rect=[0, 0.055, 1, 0.91])
    plt.savefig(output_path, dpi=300)
    print(f"Saved verified benchmark chart to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    build_chart()
