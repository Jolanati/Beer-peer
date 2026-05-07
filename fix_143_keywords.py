import json

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)

NEW_14_3 = """\
# \u2500\u2500 14.3  Auto-name clusters via TF-IDF \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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
# (non-distinctive \u2192 would bias all cluster names toward the same words)
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

print("Cluster keywords (TF-IDF) \u2014 top 15 per cluster:")
print("\u2500" * 80)
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

print(f"\\n\u2713 Section 14.3 complete \u2014 cluster_keywords and cluster_names set.")
"""

def to_source(code):
    lines = code.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            if line:
                result.append(line)
    return result

for cell in nb['cells']:
    if cell.get('id') == '1860746a':
        cell['source'] = to_source(NEW_14_3)
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Updated 14.3 cell.')
        break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Saved.')
