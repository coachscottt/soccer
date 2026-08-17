import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Style settings
matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["font.size"] = 11
sns.set_style("whitegrid")


def plot_probability_divergence(
    matches: list[dict],
    figsize: tuple = (14, 5.5),  # article used (14, 8): dead space above square panels
    save_path: str = "divergence_scatter.png",
    show: bool = True,
):
    """
    Scatter plot: bookmaker probabilities vs Polymarket.
    Points far from the diagonal = divergences = potential edge.

    matches: [{"name": "...", "bk_home": 0.55, "poly_home": 0.48}, ...]
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    outcomes = [("home", "Home Win"), ("draw", "Draw"), ("away", "Away Win")]
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

    for ax, (key, title), color in zip(axes, outcomes, colors):
        bk_probs = [m[f"bk_{key}"] for m in matches]
        poly_probs = [m[f"poly_{key}"] for m in matches]

        ax.scatter(bk_probs, poly_probs, alpha=0.6, color=color,
                   edgecolors="white", s=60)

        # Diagonal (full agreement)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)

        # Divergence zones
        ax.fill_between([0, 1], [0.05, 1.05], [0, 1],
                        alpha=0.05, color="blue",
                        label="Polymarket higher")
        ax.fill_between([0, 1], [0, 1], [-0.05, 0.95],
                        alpha=0.05, color="red",
                        label="Bookmaker higher")

        ax.set_xlabel("Bookmaker P", fontsize=11)
        ax.set_ylabel("Polymarket P", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle(
        "Probability Divergence: Bookmaker vs Polymarket\n"
        "Points far from diagonal -> potential value",
        fontsize=14, y=1.04,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_model_comparison(
    results: dict,
    save_path: str = "model_comparison.png",
    show: bool = True,
):
    """Visualization of model accuracy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results.keys())
    accuracies = [results[n]["accuracy_mean"] for n in names]
    acc_stds = [results[n]["accuracy_std"] for n in names]
    log_losses = [results[n]["log_loss_mean"] for n in names]
    ll_stds = [results[n]["log_loss_std"] for n in names]

    base_colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
    # extra rows (e.g. a market baseline) get neutral gray, never a
    # cycled repeat of a model's color
    colors = (base_colors + ["#95a5a6"] * len(names))[:len(names)]

    # Accuracy
    bars = axes[0].barh(names, accuracies, xerr=acc_stds,
                         color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Model Accuracy (TimeSeriesSplit CV)")
    axes[0].set_xlim(0.3, 0.65)
    for bar, val, std in zip(bars, accuracies, acc_stds):
        axes[0].text(val + std + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontweight="bold")

    # Log Loss
    bars = axes[1].barh(names, log_losses, xerr=ll_stds,
                         color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_xlabel("Log Loss")
    axes[1].set_title("Model Log Loss (lower = better)")
    for bar, val, std in zip(bars, log_losses, ll_stds):
        axes[1].text(val + std + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_calibration_curve(y_true, y_proba, class_idx=2,
                            class_name="Home Win",
                            save_path="calibration_curve.png",
                            show=True):
    """
    Calibration plot: shows how well
    predicted probabilities match actual outcomes.
    Ideal model — diagonal line.
    """
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(
        (np.asarray(y_true) == class_idx).astype(int),
        y_proba[:, class_idx],
        n_bins=10,
        strategy="uniform",
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", color="#e74c3c",
            label=f"Model ({class_name})", linewidth=2, markersize=8)

    ax.fill_between(prob_pred, prob_true,
                     [p for p in prob_pred],
                     alpha=0.1, color="#e74c3c")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_probability_distribution(proba, y_true,
                                  save_path="probability_distribution.png",
                                  show=True):
    """
    Visualization of predicted probability distributions
    for each class: P(class) when the class actually happened
    vs when it didn't. Good separation = informative model.
    (Article labeled these "Correct"/"Incorrect", but the split
    is by actual outcome, not prediction correctness.)
    """
    import numpy as np

    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = ["Away Win (0)", "Draw (1)", "Home Win (2)"]
    colors = ["#e74c3c", "#f1c40f", "#2ecc71"]

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        actual_mask = y_true == i
        ax.hist(proba[actual_mask, i], bins=30, alpha=0.7,
                color=color, label="Actual: this outcome", density=True)
        ax.hist(proba[~actual_mask, i], bins=30, alpha=0.3,
                color="gray", label="Actual: other", density=True)

        ax.set_xlabel(f"P({label})")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend()

    plt.suptitle("Predicted Probability Distributions",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred,
                          save_path="confusion_matrix.png",
                          show=True):
    """Confusion matrix visualization."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    labels = ["Away Win", "Draw", "Home Win"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Result", fontsize=12)
    ax.set_ylabel("Actual Result", fontsize=12)
    ax.set_title("Confusion Matrix — Ensemble Model", fontsize=14)

    # Add percentages (row-normalized: recall per class)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)
    for i in range(3):
        for j in range(3):
            ax.text(j + 0.5, i + 0.75,
                    f"({cm_pct[i, j]:.0%})",
                    ha="center", va="center",
                    fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_feature_importance(model, feature_names, top_n=15,
                            save_path="feature_importance.png",
                            show=True):
    """
    Feature importance visualization for XGBoost / Random Forest.
    """
    # For XGBoost or RF
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))

    # single-hue ramp: importance is a magnitude, not a polarity
    # (article used RdYlGn, a diverging red-green palette)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
    ax.barh(
        range(top_n),
        importances[indices],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Important Features", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_triple_layer_radar(
    match_name: str,
    bookmaker: dict,
    polymarket: dict,
    ml_model: dict,
    save_path: str = "triple_radar.png",
    show: bool = True,
):
    """
    Radar chart: comparing three probability sources
    for a single match.
    """
    categories = ["Home Win", "Draw", "Away Win"]
    keys = ["home", "draw", "away"]

    fig, ax = plt.subplots(figsize=(8, 8),
                            subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(categories),
                          endpoint=False).tolist()
    angles += angles[:1]

    # Article used blue/red/green; red+green is not CVD-safe for two
    # overlapping polygons, so ML is orange and each source gets its
    # own line style (identity is not carried by color alone).
    sources = [
        ("Bookmaker", bookmaker, "#3498db", "-"),
        ("Polymarket", polymarket, "#e74c3c", "--"),
        ("ML Model", ml_model, "#f39c12", ":"),
    ]

    for label, probs, color, style in sources:
        values = [probs[k] for k in keys]
        values += values[:1]
        ax.plot(angles, values, "o", linestyle=style, linewidth=2,
                label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 0.8)
    ax.set_title(f"Triple Layer: {match_name}",
                 fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
