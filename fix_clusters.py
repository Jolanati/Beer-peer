import json

# ── Apply 3 cluster fixes to wine-dine.ipynb ────────────────────────────────
# Fix 1: TF-IDF quality-word stoplist  (§14.3)
# Fix 2: Top-2 keyword cluster names   (§14.3)
# Fix 3: BisectingKMeans final fit     (§14.2)

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)


def to_source(code: str):
    """Split a plain string into the notebook source list format."""
    lines = code.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            if line:          # omit a trailing empty line
                result.append(line)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# §14.2 — Replace standard KMeans final fit with BisectingKMeans
# ─────────────────────────────────────────────────────────────────────────────
NEW_14_2 = """\
# ── 14.2  K-means clustering with silhouette justification ───────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

K_RANGE    = range(6, 21)
K_CLUSTERS = 12
N_SIL      = min(5_000, len(train_taste_vecs))     # silhouette subsample

# ── Silhouette curve (standard KMeans, for comparability) ────────────────────
rng      = np.random.default_rng(42)
sil_idx  = rng.choice(len(train_taste_vecs), N_SIL, replace=False)
vecs_sil = train_taste_vecs[sil_idx]

sil_scores = []
print("Computing silhouette scores …")
for k in K_RANGE:
    _km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=200)
    _lb = _km.fit_predict(vecs_sil)
    s   = silhouette_score(vecs_sil, _lb, metric="cosine",
                           sample_size=2_000, random_state=42)
    sil_scores.append(s)
    print(f"  K={k:2d}  silhouette={s:.4f}")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(list(K_RANGE), sil_scores, marker="o", linewidth=1.5, color="#4C72B0")
ax.axvline(K_CLUSTERS, color="#C44E52", linestyle="--", label=f"Selected K={K_CLUSTERS}")
ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Silhouette score (cosine)")
ax.set_title("K-means silhouette score — TasteBiLSTM embedding space")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, "kmeans_silhouette.png")
plt.close(fig)

# ── Final BisectingKMeans fit ─────────────────────────────────────────────────
# BisectingKMeans repeatedly splits the largest cluster rather than fitting
# all centroids simultaneously.  This yields far more balanced cluster sizes
# while preserving tight intra-cluster cohesion (nearly same silhouette score).
print(f"\\nFitting final BisectingKMeans  K={K_CLUSTERS}  on {len(train_taste_vecs):,} vectors …")
kmeans         = BisectingKMeans(n_clusters=K_CLUSTERS, random_state=42,
                                  n_init=3, max_iter=300)
cluster_labels = kmeans.fit_predict(train_taste_vecs)          # (N_train,)
centroids      = normalize(kmeans.cluster_centers_, norm="l2") # (K, 512)

counts = np.bincount(cluster_labels, minlength=K_CLUSTERS)
print(f"Inertia : {kmeans.inertia_:.2f}")
print("\\nCluster sizes:")
for k in range(K_CLUSTERS):
    bar = "█" * max(1, counts[k] * 30 // counts.max())
    print(f"  {k:2d}: {counts[k]:6,}  {bar}")

print(f"\\n✓ Section 14.2 complete — BisectingKMeans K={K_CLUSTERS} fitted; centroids shape {centroids.shape}.")
"""

# ─────────────────────────────────────────────────────────────────────────────
# §14.3 — Better stopwords + top-2 keyword names
# ─────────────────────────────────────────────────────────────────────────────
NEW_14_3 = """\
# ── 14.3  Auto-name clusters via TF-IDF ──────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

# Build one document per cluster (all review texts concatenated)
cluster_docs = []
for k in range(K_CLUSTERS):
    mask  = cluster_labels == k
    texts = df_train.loc[mask, "review_text"].tolist()
    cluster_docs.append(" ".join(texts))

tfidf        = TfidfVectorizer(max_features=3_000, stop_words="english", min_df=2)
tfidf_matrix = tfidf.fit_transform(cluster_docs)   # (K, V)
feat_names   = tfidf.get_feature_names_out()

# Structural wine vocab + review-quality words that appear in every cluster
# (non-distinctive → would bias all cluster names toward the same words)
_SKIP_KW = {
    "wine", "wines", "drink", "palate", "aroma", "aromas",
    "finish", "flavor", "flavors", "notes", "nose",
    "taste", "tastes", "bottle", "glass",
    # evaluation / quality language (not taste signal)
    "good", "great", "nice", "smooth", "really", "well",
    "just", "very", "also", "lot", "quite", "bit",
    "little", "long", "short", "high", "low", "like",
}

cluster_keywords = {}
cluster_names    = {}

print("Cluster keywords (TF-IDF):")
print("─" * 68)
for k in range(K_CLUSTERS):
    row    = tfidf_matrix[k].toarray()[0]
    ranked = sorted(range(len(feat_names)), key=lambda i: row[i], reverse=True)
    kws    = [feat_names[i] for i in ranked if feat_names[i] not in _SKIP_KW][:5]
    cluster_keywords[k] = kws[:3]
    # Top-2 keywords give unambiguous names even when two clusters share kw[0]
    if len(kws) >= 2:
        cluster_names[k] = f"{kws[0].title()} / {kws[1].title()}"
    elif kws:
        cluster_names[k] = kws[0].title()
    else:
        cluster_names[k] = f"Cluster {k}"
    print(f"  {k:2d}  {cluster_names[k]:<30}  ·  {kws}")

print(f"\\n✓ Section 14.3 complete — cluster_keywords and cluster_names set.")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Patch the notebook cells
# ─────────────────────────────────────────────────────────────────────────────
TARGET_14_2 = "8839e43f"
TARGET_14_3 = "1860746a"

changed = 0
for cell in nb['cells']:
    cid = cell.get('id', '')
    if cid == TARGET_14_2:
        cell['source'] = to_source(NEW_14_2)
        cell['outputs'] = []
        cell['execution_count'] = None
        changed += 1
        print(f"  ✓ §14.2 cell ({cid}) updated — BisectingKMeans")
    elif cid == TARGET_14_3:
        cell['source'] = to_source(NEW_14_3)
        cell['outputs'] = []
        cell['execution_count'] = None
        changed += 1
        print(f"  ✓ §14.3 cell ({cid}) updated — stopwords + top-2 names")

if changed == 2:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"\nNotebook saved. {changed}/2 cells patched.")
else:
    print(f"\nERROR: expected 2 cells, found {changed}. Notebook NOT saved.")
    print("Cell IDs in notebook:", [c.get('id') for c in nb['cells']])
