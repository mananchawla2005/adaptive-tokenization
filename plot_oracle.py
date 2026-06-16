import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 96-token data
f96_crs =   [0.03, 0.06, 0.14, 0.17, 0.21, 0.25, 0.32, 0.35, 0.41, 0.49, 0.52, 0.58, 0.61, 0.66, 0.71]
f96_losses =[6.213,6.167,6.182,6.196,6.209,6.231,6.243,6.253,6.326,6.309,6.343,6.240,6.161,6.170,6.223]
f96_naive  = 6.322
f96_adapt  = 6.221

# 384-token data
f384_crs   = [0.04, 0.05, 0.13, 0.19, 0.24, 0.29, 0.31, 0.35, 0.41, 0.46, 0.55, 0.56, 0.61, 0.66, 0.70]
f384_losses=[2.185,2.170,2.156,2.119,2.104,2.128,2.103,2.110,2.149,2.123,2.118,2.138,2.665,3.943,4.152]
f384_naive = 2.046
f384_adapt = 2.217

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
            ax.axvspan(cr-0.02, cr+0.02, alpha=0.08, color="green")
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
    ax2t.set_xticklabels([f"{int(prompt_len*(1-t))}" for t in [0.0, 0.2, 0.4, 0.6]])
    ax2t.set_xlabel("Span count", fontsize=10)

plot_oracle(ax1, f96_crs, f96_losses, f96_naive, f96_adapt, 96, "96-token Prompt")
plot_oracle(ax2, f384_crs, f384_losses, f384_naive, f384_adapt, 157, "157-token Prompt")

plt.suptitle("Adaptive Tokenization Oracle: Adaptive + Compression vs Naive Baseline", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("oracle_dual.png", dpi=150, bbox_inches="tight")
print("Saved oracle_dual.png")
