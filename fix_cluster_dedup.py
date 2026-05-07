import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)

# ─── Fix A: K_CLUSTERS 10 → 9 (cell 8839e43f) ───────────────────────────────
FIX_A_OLD = 'K_CLUSTERS = 10  # 10 matches the 10 TasteBiLSTM axes; BisectingKMeans makes 9 splits'
FIX_A_NEW = 'K_CLUSTERS = 9   # 9 clusters: 10 taste axes minus 1 "garbage/balanced" catchall'

# ─── Fix B: §14.3 full replacement (cell 1860746a) ───────────────────────────
NEW_143 = r'''# ── 14.3  Auto-name clusters via TF-IDF + taste-axis naming ─────────────────
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

# ── Words to exclude from TF-IDF display ─────────────────────────────────────
_SKIP_KW = {
    # structural wine-review words (no flavor content)
    "wine", "wines", "drink", "drinking", "palate", "aroma", "aromas",
    "finish", "flavor", "flavors", "notes", "nose",
    "taste", "tastes", "bottle", "glass",
    # quality / evaluation language
    "good", "great", "nice", "smooth", "really", "well",
    "just", "very", "also", "lot", "quite", "bit",
    "little", "long", "short", "high", "low", "like",
    # sentiment / review praise (cluster 9 garbage words)
    "delicious", "perfect", "love", "amazing", "beautiful",
    "wonderful", "lovely", "excellent", "fantastic",
    # price / value / recommendation language
    "price", "value", "best", "easy",
    "bold", "strong", "balanced", "buy", "money",
    # grape / geographic / stylistic labels (not sensory)
    "pinot", "cabernet", "merlot", "syrah", "chardonnay",
    "italian", "french", "spanish", "american", "style",
    "region", "grape", "varietal", "variety", "blend",
    # temporal / generic descriptors
    "year", "years", "age", "aging", "made", "make",
    "red", "white", "full", "medium", "light",
}

cluster_keywords = {}
cluster_names    = {}

# ── Step 1: TF-IDF keywords (diagnostic only) ────────────────────────────────
print("Cluster TF-IDF keywords (diagnostic — top 10 per cluster):")
print("─" * 80)
for k in range(K_CLUSTERS):
    row    = tfidf_matrix[k].toarray()[0]
    ranked = sorted(range(len(feat_names)), key=lambda i: row[i], reverse=True)
    all_kws = [(feat_names[i], round(row[i], 4)) for i in ranked
               if feat_names[i] not in _SKIP_KW][:10]
    kws = [w for w, _ in all_kws]
    cluster_keywords[k] = kws[:3]
    kw_str = ", ".join(f"{w}({s})" for w, s in all_kws)
    print(f"  {k:2d}  {kw_str}")

# ── Step 2: Taste-axis naming with deduplication ─────────────────────────────
#
# Algorithm:
#   1. Compute mean taste scores per cluster.
#   2. Clusters with all scores < 0.05 → "The Balanced" (generic/noise cluster).
#   3. For the rest, assign names greedily by dominance (strongest cluster first).
#   4. Deduplication: if a name is already taken, the duplicate cluster skips
#      any axes already used in the conflicting name and picks its next-best
#      unique secondary axis — so two cherry clusters become
#      "The Berry & The Smoky" vs "The Berry & The Dark" etc.

_taste_train_t = torch.tensor(taste_train, dtype=torch.float32)  # (N, 10)

# Compute mean scores for all clusters up front
_cluster_means = {}
for k in range(K_CLUSTERS):
    mask = cluster_labels == k
    if mask.sum() == 0:
        _cluster_means[k] = torch.zeros(len(TASTE_AXES))
    else:
        _cluster_means[k] = _taste_train_t[mask].mean(dim=0)

# Sort clusters by their top-axis score descending — strongest gets first pick
_cluster_order = sorted(range(K_CLUSTERS),
                        key=lambda k: _cluster_means[k].max().item(),
                        reverse=True)

_used_names = set()

for k in _cluster_order:
    mean_scores = _cluster_means[k]
    top_score   = mean_scores.max().item()

    # Garbage / balanced cluster: no dominant axis
    if top_score < 0.05:
        cluster_names[k] = "The Balanced"
        continue

    ranked_axes = mean_scores.argsort(descending=True).tolist()

    # Try combinations of top axes until we find a unique name
    name = None
    for i in range(len(ranked_axes)):
        ax1   = TASTE_AXES[ranked_axes[i]]
        s1    = mean_scores[ranked_axes[i]].item()
        name1 = TASTE_AXIS_NAMES[ax1]

        # Single-axis name (if dominant)
        s2 = mean_scores[ranked_axes[i+1]].item() if i+1 < len(ranked_axes) else 0
        if s1 > 0 and (s2 == 0 or s2 / s1 < 0.5):
            candidate = name1
            if candidate not in _used_names:
                name = candidate
                break
            # taken — fall through to two-axis

        # Two-axis name: pair with next available axis not already in a conflict
        for j in range(i+1, len(ranked_axes)):
            ax2       = TASTE_AXES[ranked_axes[j]]
            candidate = f"{TASTE_AXIS_NAMES[ax1]} & {TASTE_AXIS_NAMES[ax2]}"
            if candidate not in _used_names:
                name = candidate
                break
        if name:
            break

    cluster_names[k] = name if name else f"Cluster {k}"
    _used_names.add(cluster_names[k])

# ── Print final names in cluster order ───────────────────────────────────────
print("\n── Taste-axis cluster names ─────────────────────────────────────────────────")
print(f"  {'#':>2}  {'Name':<34}  {'Top axes (mean score)'}")
print("  " + "─" * 74)
for k in range(K_CLUSTERS):
    mean_scores = _cluster_means[k]
    ranked_axes = mean_scores.argsort(descending=True).tolist()
    axis_str = "  ".join(
        f"{TASTE_AXIS_NAMES[TASTE_AXES[i]]}: {mean_scores[i].item():.3f}"
        for i in ranked_axes[:4]
    )
    print(f"  {k:2d}  {cluster_names[k]:<34}  {axis_str}")

print(f"\n✓ Section 14.3 complete — cluster_keywords and cluster_names set.")
'''

def to_source(code):
    lines = code.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        elif line:
            result.append(line)
    return result

changed = 0
for cell in nb['cells']:
    cid = cell.get('id', '')
    src = ''.join(cell.get('source', []))

    if cid == '8839e43f' and FIX_A_OLD in src:
        cell['source'] = to_source(src.replace(FIX_A_OLD, FIX_A_NEW))
        cell['outputs'] = []
        cell['execution_count'] = None
        changed += 1
        print('  ✓ §14.2 K_CLUSTERS 10 → 9')

    elif cid == '1860746a':
        cell['source'] = to_source(NEW_143)
        cell['outputs'] = []
        cell['execution_count'] = None
        changed += 1
        print('  ✓ §14.3 full replacement (stoplist + dedup naming)')

if changed == 2:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'\nSaved. {changed}/2 changes applied.')
else:
    print(f'\nWARNING: only {changed}/2 applied.')
