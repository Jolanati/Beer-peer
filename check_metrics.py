import json

# ── Apply 3 cluster fixes to wine-dine.ipynb ────────────────────────────────
# Fix 1: TF-IDF quality-word stoplist  (§14.3)
# Fix 2: Top-2 keyword cluster names   (§14.3)
# Fix 3: BisectingKMeans final fit     (§14.2)

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)

OLD_FIRST_LINE = "# \u2500\u2500 14.4  Representative wine per cluster"
NEW_SOURCE = [
    "# \u2500\u2500 14.4  Representative wine per cluster \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "# Top 20 % by rating_pct \u2192 from those pick review nearest to cluster centroid.\n",
    "# Wine labels are deduplicated across clusters: once a label is claimed by an\n",
    "# earlier cluster it is skipped for all subsequent ones.\n",
    "\n",
    "cluster_wines = {}   # k \u2192 {\"wine_label\": str, \"review_text\": str, \"rating_pct\": int}\n",
    "used_labels   = set()  # tracks wine_labels already assigned to an earlier cluster\n",
    "\n",
    "for k in range(K_CLUSTERS):\n",
    "    mask    = cluster_labels == k\n",
    "    sub_idx = np.where(mask)[0]            # original integer positions in df_train / X_train\n",
    "    sub_df  = df_train.iloc[sub_idx].copy()\n",
    "\n",
    "    # Top 20 % threshold (at least 1 review)\n",
    "    thresh  = sub_df[\"rating_pct\"].quantile(0.80)\n",
    "    top_df  = sub_df[sub_df[\"rating_pct\"] >= thresh]\n",
    "    if top_df.empty:\n",
    "        top_df = sub_df\n",
    "\n",
    "    # Rank all top-rated candidates by similarity to centroid (best first)\n",
    "    top_pos = np.array([i for i, row_idx in zip(sub_idx, sub_df.index)\n",
    "                        if row_idx in top_df.index])\n",
    "    v_top   = train_taste_vecs[top_pos]   # (M, 512)\n",
    "    sims    = v_top @ centroids[k]        # (M,)\n",
    "    ranked  = top_pos[np.argsort(-sims)]  # descending similarity\n",
    "\n",
    "    # Walk down ranked list; pick first candidate whose label hasn't been used yet.\n",
    "    # Fall back to the best-similarity pick if every candidate label is already taken.\n",
    "    chosen = ranked[0]\n",
    "    for pos in ranked:\n",
    "        lbl = (str(df_train.iloc[pos].get(\"wine_label\", \"\")).strip()\n",
    "               or str(df_train.iloc[pos].get(\"grape_class\", \"\u2014\")))\n",
    "        if lbl not in used_labels:\n",
    "            chosen = pos\n",
    "            break\n",
    "\n",
    "    row        = df_train.iloc[chosen]\n",
    "    wine_label = (str(row.get(\"wine_label\", \"\")).strip()\n",
    "                  or str(row.get(\"grape_class\", \"\u2014\")))\n",
    "    used_labels.add(wine_label)\n",
    "\n",
    "    cluster_wines[k] = {\n",
    "        \"wine_label\"  : wine_label,\n",
    "        \"review_text\" : str(row[\"review_text\"])[:180],\n",
    "        \"rating_pct\"  : int(row[\"rating_pct\"]),\n",
    "    }\n",
    "\n",
    "print(\"Representative wine per cluster:\")\n",
    "print(\"\u2500\" * 72)\n",
    "for k in range(K_CLUSTERS):\n",
    "    w = cluster_wines[k]\n",
    "    print(f\"  {k:2d}  {cluster_names[k]:<32}  {w['wine_label']:<28}  {w['rating_pct']}%\")\n",
    "\n",
    "# Sanity-check: warn if any label was duplicated (fallback path was hit)\n",
    "all_lbls = [cluster_wines[k][\"wine_label\"] for k in range(K_CLUSTERS)]\n",
    "dups = [lbl for lbl in set(all_lbls) if all_lbls.count(lbl) > 1]\n",
    "if dups:\n",
    "    print(f\"\\n\u26a0 Duplicate labels (fallback triggered): {dups}\")\n",
    "else:\n",
    "    print(\"\\n\u2713 All representative wines are unique across clusters.\")\n",
    "print(f\"\u2713 Section 14.4 complete \u2014 cluster_wines populated for all {K_CLUSTERS} clusters.\")\n",
]

found = False
for cell in nb['cells']:
    src = cell.get('source', [])
    joined = ''.join(src)
    if OLD_FIRST_LINE in joined and 'best_pos = top_pos[int(np.argmax(sims))]' in joined:
        cell['source'] = NEW_SOURCE
        cell['outputs'] = []
        cell['execution_count'] = None
        found = True
        print("Cell updated.")
        break

if not found:
    print("ERROR: target cell not found.")
else:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Notebook saved.")


# Check structure of a cell with known output (cell index 2)
cell = nb['cells'][2]
print("Cell 2 structure keys:", list(cell.keys()))
print("Cell 2 id:", cell.get('id'))
outs = cell.get('outputs', [])
print("Outputs count:", len(outs))
if outs:
    for j, out in enumerate(outs):
        print(f"  Output {j}: keys={list(out.keys())}, type={out.get('output_type')}")
        for k, v in out.items():
            if k != 'text':
                print(f"    {k}: {repr(v)[:100]}")
        if 'text' in out:
            txt = ''.join(out['text'])
            print(f"    text (first 300): {repr(txt[:300])}")

