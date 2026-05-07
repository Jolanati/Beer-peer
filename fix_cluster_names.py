import json

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)

OLD = '''cluster_keywords = {}
cluster_names    = {}

print("Cluster keywords (TF-IDF) — top 15 per cluster:")
print("─" * 80)
for k in range(K_CLUSTERS):
    row    = tfidf_matrix[k].toarray()[0]
    ranked = sorted(range(len(feat_names)), key=lambda i: row[i], reverse=True)
    # top-15 for display (skipping structural words)
    all_kws = [(feat_names[i], round(row[i], 4)) for i in ranked
               if feat_names[i] not in _SKIP_KW][:15]
    kws = [w for w, _ in all_kws]
    cluster_keywords[k] = kws[:3]
    # Top-2 keywords give unambiguous names even when two clusters share kw[0]
    if len(kws) >= 2:
        cluster_names[k] = f"{kws[0].title()} / {kws[1].title()}"
    elif kws:
        cluster_names[k] = kws[0].title()
    else:
        cluster_names[k] = f"Cluster {k}"
    kw_str = ", ".join(f"{w}({s})" for w, s in all_kws)
    print(f"  {k:2d}  {cluster_names[k]:<28}  {kw_str}")

print(f"\\n✓ Section 14.3 complete — cluster_keywords and cluster_names set.")'''

NEW = '''cluster_keywords = {}
cluster_names    = {}

# ── Step 1: TF-IDF keywords (kept for diagnostics / debugging) ────────────────
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

# ── Step 2: Taste-axis naming — user-friendly names from TASTE_AXIS_NAMES ─────
#
# For each cluster: compute the mean taste score per axis across all member wines.
# The top-2 dominant axes determine the cluster name.
# This is more meaningful than TF-IDF keywords because it uses the actual model
# scores rather than raw word frequency.
#
# Name format:  "The Berry & The Grippy"  (top-2 axes)
#               "The Smoky"               (if one axis dominates clearly)

print("\\n── Taste-axis cluster names ─────────────────────────────────────────────────")
print(f"  {'#':>2}  {'Name':<32}  {'Top axes (mean score)'}")
print("  " + "─" * 72)

_taste_train_t = torch.tensor(taste_train, dtype=torch.float32)  # (N, 10)

for k in range(K_CLUSTERS):
    mask        = cluster_labels == k
    if mask.sum() == 0:
        cluster_names[k] = f"Cluster {k}"
        continue

    # Mean taste-axis score for all wines in this cluster
    mean_scores = _taste_train_t[mask].mean(dim=0)  # (10,)

    # Rank axes by mean score — pick top-2
    ranked_axes = mean_scores.argsort(descending=True)
    ax1 = TASTE_AXES[ranked_axes[0].item()]
    ax2 = TASTE_AXES[ranked_axes[1].item()]
    s1  = mean_scores[ranked_axes[0]].item()
    s2  = mean_scores[ranked_axes[1]].item()

    # If the top axis is much stronger than the second, use single name
    if s1 > 0 and s2 / s1 < 0.5:
        cluster_names[k] = TASTE_AXIS_NAMES[ax1]
    elif s1 > 0:
        cluster_names[k] = f"{TASTE_AXIS_NAMES[ax1]} & {TASTE_AXIS_NAMES[ax2]}"
    else:
        cluster_names[k] = f"Cluster {k}"

    axis_str = "  ".join(
        f"{TASTE_AXIS_NAMES[TASTE_AXES[i.item()]]}: {mean_scores[i].item():.3f}"
        for i in ranked_axes[:4]
    )
    print(f"  {k:2d}  {cluster_names[k]:<32}  {axis_str}")

print(f"\\n✓ Section 14.3 complete — cluster_keywords and cluster_names set.")'''

changed = 0
for cell in nb['cells']:
    if cell.get('id') == '1860746a':
        src = ''.join(cell.get('source', []))
        if OLD in src:
            new_src = src.replace(OLD, NEW)
            lines = new_src.split('\n')
            result = []
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    result.append(line + '\n')
                elif line:
                    result.append(line)
            cell['source'] = result
            cell['outputs'] = []
            cell['execution_count'] = None
            changed += 1
            print('  ✓ §14.3 naming logic updated')
        else:
            print('  ✗ OLD string not found in §14.3')

if changed == 1:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Saved.')
else:
    print('NOT saved.')
