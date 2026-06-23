import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 96-token data (from oracle run 2026-06-22, bugs fixed)
f96_crs =    [0.01, 0.05, 0.15, 0.17, 0.21, 0.29, 0.34, 0.39, 0.43, 0.47, 0.51, 0.57, 0.61, 0.66]
f96_losses = [5.301, 5.302, 5.312, 5.300, 5.289, 5.269, 5.246, 5.223, 5.186, 5.205, 5.002, 4.916, 5.017, 5.067]
f96_naive  = 5.2564
f96_adapt  = 5.3319

# 384-token data (from oracle run 2026-06-22, bugs fixed)
f384_crs =    [0.04, 0.07, 0.15, 0.18, 0.20, 0.29, 0.31, 0.37, 0.42, 0.50, 0.52, 0.59, 0.61, 0.65]
f384_losses = [4.443, 4.429, 4.446, 4.420, 4.361, 4.380, 4.412, 4.454, 4.448, 4.350, 4.195, 3.988, 3.903, 4.486]
f384_naive = 4.4918
f384_adapt = 4.4991

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

def plot_oracle(ax, crs, losses, naive_loss, adapt_loss, prompt_len, title):
    adapt_best = min(losses)
    ax.plot(crs, losses, "r-o", linewidth=2.5, markersize=6, label="Adaptive Pareto frontier")
    ax.axhline(y=adapt_loss, color="red", linestyle="--", linewidth=1.8, alpha=0.7,
               label=f"Adapt no-merge ({adapt_loss:.3f})")
    ax.axhline(y=naive_loss, color="blue", linestyle="--", linewidth=1.8, alpha=0.7,
               label=f"Naive no-merge ({naive_loss:.3f})")

    # Green zone: adaptive+compression beats naive
    for cr, l in zip(crs, losses):
        if l < naive_loss:
            ax.axvspan(cr - 0.02, cr + 0.02, alpha=0.08, color="green")
    ax.axvspan(0, 1, alpha=0.04, color="green", label="Adaptive+compression < Naive no-merge")

    best_idx = losses.index(adapt_best)
    gain = (naive_loss - adapt_best) / naive_loss * 100
    ax.annotate(f"Best: cr={crs[best_idx]:.0%}\n{gain:.1f}% better than naive",
                xy=(crs[best_idx], adapt_best),
                xytext=(0.4, adapt_best - (naive_loss - adapt_best) * 0.3),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5), color="darkred")

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.75)

    ax2t = ax.twiny()
    ax2t.set_xlim(ax.get_xlim())
    ax2t.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax2t.set_xticklabels([f"{int(prompt_len * (1 - t))}" for t in [0.0, 0.2, 0.4, 0.6]])
    ax2t.set_xlabel("Span count", fontsize=10)

plot_oracle(ax1, f96_crs, f96_losses, f96_naive, f96_adapt, 96, "96-token Prompt")
plot_oracle(ax2, f384_crs, f384_losses, f384_naive, f384_adapt, 384, "384-token Prompt")

plt.suptitle("Adaptive Tokenization Oracle: Adaptive + Compression vs Naive Baseline", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("oracle_dual.png", dpi=150, bbox_inches="tight")
print("Saved oracle_dual.png")
