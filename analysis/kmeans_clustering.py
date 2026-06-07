"""
Global Election Systems: K-Means Clustering Analysis
=====================================================
Research focus: California (USA) — positioned against global comparators
on two dimensions:
  X  count_score  — ballot counting window (1=same night, 10=unresolved/permanent dispute)
  Y  id_score     — voter ID strictness    (1=none, 10=biometric)

Algorithm: K-Means++ (k=3), with semantic label remapping so that
  Cluster 2 (green)  = mature democracies  (low X, high Y)
  Cluster 1 (yellow) = transitional systems (mid X, mid Y)
  Cluster 0 (red)    = fragile systems      (high X, low Y)

Usage:
    pip install pandas scikit-learn matplotlib
    python analysis/kmeans_clustering.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ── 0. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/election_systems.csv")
X = df[["count_score", "id_score"]].values

# ── 1. K-Means++ clustering (k=3) ────────────────────────────────────────────
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=50, random_state=42)
kmeans.fit(X)
df["raw_cluster"] = kmeans.labels_
centers = kmeans.cluster_centers_

# ── 2. Semantic remapping ─────────────────────────────────────────────────────
scores = centers[:, 1] - centers[:, 0]
rank = np.argsort(scores)[::-1]
remap = {rank[0]: 2, rank[1]: 1, rank[2]: 0}
df["cluster"] = df["raw_cluster"].map(remap)

CLUSTER_NAMES  = {2: "Cluster 1: Mature Democracies",
                  1: "Cluster 2: Transitional Systems",
                  0: "Cluster 3: Fragile / Disputed"}
CLUSTER_COLORS = {2: "#06d6a0", 1: "#ffd166", 0: "#ff6b6b"}
CA_COLOR = "#ff9f1c"

# ── 3. Print summary ──────────────────────────────────────────────────────────
print("=" * 60)
print("GLOBAL ELECTION SYSTEMS — K-MEANS CLUSTERING RESULTS")
print("=" * 60)
for cid in [2, 1, 0]:
    orig = [k for k, v in remap.items() if v == cid][0]
    cx, cy = centers[orig]
    members = df[df["cluster"] == cid]["country"].tolist()
    print(f"\n{'─'*40}")
    print(f"{CLUSTER_NAMES[cid]}")
    print(f"  Centroid → count={cx:.2f}, id={cy:.2f}")
    print(f"  Members  → {', '.join(members)}")

ca = df[df["focus"] == 1].iloc[0]
print(f"\n{'='*60}")
print(f"★ FOCUS — California (USA)")
print(f"   count_score = {ca['count_score']}  |  id_score = {ca['id_score']}")
print(f"   Cluster     = {CLUSTER_NAMES[ca['cluster']]}")
print("=" * 60)

# ── 4. Label offset map (hand-tuned to avoid overlap) ────────────────────────
OFFSETS = {
    # Western Europe cluster (dense top-left) — spread them out
    "Germany":        (-0.25, -0.45),
    "France":         ( 0.15,  0.20),
    "Netherlands":    ( 0.15, -0.45),
    "Sweden":         (-0.25,  0.22),
    "Norway":         (-1.60,  0.22),
    "Belgium":        (-1.85, -0.42),
    "Spain":          (-0.25, -0.72),
    "Italy":          ( 0.15,  0.42),
    "Switzerland":    ( 0.15, -0.68),
    "Austria":        (-1.80, -0.20),
    "Denmark":        (-1.60, -0.68),
    "Finland":        ( 0.15,  0.64),
    "Portugal":       (-1.90,  0.42),
    "Ireland":        ( 0.15,  0.20),
    # Asia / others
    "Japan":          ( 0.15, -0.40),
    "South Korea":    ( 0.15,  0.22),
    "Singapore":      ( 0.15,  0.22),
    "Taiwan":         ( 0.15, -0.40),
    "Israel":         ( 0.15, -0.40),
    "Canada":         ( 0.15,  0.22),
    "United Kingdom": ( 0.15,  0.22),
    # Transitional
    "California (USA)": ( 0.20,  0.28),
    "Colombia":         ( 0.15,  0.22),
    "Panama":           ( 0.15, -0.40),
    "Guinea":           ( 0.15, -0.40),
    "Peru":             ( 0.15,  0.22),
    # Fragile
    "Venezuela":     (-0.20,  0.22),
    "Nicaragua":     ( 0.15,  0.22),
    "Haiti":         (-0.20, -0.40),
    "Bolivia":       (-0.80,  0.22),
    "Honduras":      ( 0.15, -0.40),
    "Guatemala":     ( 0.15,  0.22),
    "DR Congo":      ( 0.15,  0.22),
    "Somalia":       (-0.20, -0.40),
    "Sudan":         ( 0.15,  0.22),
    "Mali":          (-0.80, -0.40),
}

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 10))
fig.patch.set_facecolor("#0b0e14")
ax.set_facecolor("#131720")

# Grid
for v in range(0, 11, 2):
    ax.axvline(v, color="#1e2535", linewidth=0.8, zorder=0)
    ax.axhline(v, color="#1e2535", linewidth=0.8, zorder=0)

# Data points + labels (no halos)
for _, row in df.iterrows():
    is_ca = row["focus"] == 1
    col   = CA_COLOR if is_ca else CLUSTER_COLORS[row["cluster"]]
    size  = 130 if is_ca else 55
    zord  = 6  if is_ca else 4

    ax.scatter(row["count_score"], row["id_score"],
               color=col, s=size, zorder=zord,
               edgecolors="white" if is_ca else col,
               linewidths=2.0 if is_ca else 0.6,
               alpha=0.95)

    name = row["country"]
    ox, oy = OFFSETS.get(name, (0.15, 0.22))
    ha = "right" if ox < 0 else "left"

    if is_ca:
        # star marker above dot
        ax.plot(row["count_score"], row["id_score"] + 0.55,
                marker="*", color=CA_COLOR, markersize=10, zorder=7)
        ax.annotate(name,
                    xy=(row["count_score"], row["id_score"]),
                    xytext=(row["count_score"] + ox,
                            row["id_score"]   + oy),
                    color=CA_COLOR, fontsize=11, fontweight="bold",
                    ha=ha, va="bottom", zorder=7)
    else:
        ax.annotate(name,
                    xy=(row["count_score"], row["id_score"]),
                    xytext=(row["count_score"] + ox,
                            row["id_score"]   + oy),
                    color=col, fontsize=9.2, ha=ha, va="bottom",
                    zorder=5, alpha=0.92)

# Axes
ax.set_xlim(-0.4, 10.6)
ax.set_ylim(-0.4, 10.6)
ax.set_xticks(range(0, 11, 2))
ax.set_yticks(range(0, 11, 2))
ax.tick_params(colors="#5a6580", labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor("#1e2535")

ax.set_xlabel("Ballot Counting Window Score  (1 = same night  →  10 = permanent dispute)",
              color="#8090b0", fontsize=11, labelpad=12)
ax.set_ylabel("Voter ID Strictness Score  (1 = none  →  10 = biometric)",
              color="#8090b0", fontsize=11, labelpad=12)
ax.set_title("Global Election Systems: K-Means Clustering (k = 3)\n"
             "Counting Speed  ×  Voter ID Strictness  ·  n = 34",
             color="#d4dbe8", fontsize=14, fontweight="bold", pad=18)

# Legend
patches = [mpatches.Patch(color=CLUSTER_COLORS[c], label=CLUSTER_NAMES[c])
           for c in [2, 1, 0]]
patches.append(mpatches.Patch(color=CA_COLOR,
               label="★  California (USA) — Research Focus"))
ax.legend(handles=patches, loc="lower left",
          facecolor="#1a2030", edgecolor="#2e3a56",
          labelcolor="#d4dbe8", fontsize=9.5,
          framealpha=0.92, borderpad=1.0)

# Rubric (top-right)
rubric = ("Scoring Rubric\n"
          "Count:  1 = same night · 3 = 1-2 days\n"
          "        5 = 5-7 days   · 7 = 2-3 weeks · 9 = 30+ days\n"
          "ID:     1 = none       · 3 = roll check\n"
          "        7 = non-photo  · 9 = mandatory photo ID")
ax.text(10.5, 10.5, rubric, color="#5a6580", fontsize=7.5,
        va="top", ha="right", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#0d1018",
                  ec="#1e2535", lw=0.8))

plt.tight_layout()
plt.savefig("figures/cluster_plot.png", dpi=180,
            bbox_inches="tight", facecolor="#0b0e14")
print("\nFigure saved → figures/cluster_plot.png")
plt.show()
