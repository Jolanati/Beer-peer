import json

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)

FIXES = {
    # ── §14.2: lower K from 12 to 8 ─────────────────────────────────────────
    '8839e43f': [
        (
            'K_CLUSTERS = 12',
            'K_CLUSTERS = 8   # 8 matches the 8 TasteBiLSTM axes; 12 over-splits into duplicate clusters',
        ),
    ],
    # ── §14.3: expand _SKIP_KW ───────────────────────────────────────────────
    '1860746a': [
        (
            '''\
_SKIP_KW = {
    "wine", "wines", "drink", "palate", "aroma", "aromas",
    "finish", "flavor", "flavors", "notes", "nose",
    "taste", "tastes", "bottle", "glass",
    # evaluation / quality language (not taste signal)
    "good", "great", "nice", "smooth", "really", "well",
    "just", "very", "also", "lot", "quite", "bit",
    "little", "long", "short", "high", "low", "like",
}''',
            '''\
_SKIP_KW = {
    # structural wine-review words (no flavor content)
    "wine", "wines", "drink", "palate", "aroma", "aromas",
    "finish", "flavor", "flavors", "notes", "nose",
    "taste", "tastes", "bottle", "glass",
    # quality / evaluation language
    "good", "great", "nice", "smooth", "really", "well",
    "just", "very", "also", "lot", "quite", "bit",
    "little", "long", "short", "high", "low", "like",
    # price / value / recommendation language
    "price", "value", "best", "excellent", "easy",
    "bold", "strong", "balanced", "buy", "money",
    # geographic / stylistic labels (not sensory)
    "italian", "french", "spanish", "american", "style",
    "region", "grape", "varietal", "variety", "blend",
    # temporal / generic descriptors
    "year", "years", "age", "aging", "made", "make",
    "red", "white", "full", "medium", "light",
}''',
        ),
    ],
}

changed = 0
for cell in nb['cells']:
    cid = cell.get('id', '')
    if cid in FIXES:
        src = ''.join(cell.get('source', []))
        for old, new in FIXES[cid]:
            if old in src:
                src = src.replace(old, new)
                changed += 1
                print(f'  ✓ patched cell {cid}: {repr(old[:50])}...')
            else:
                print(f'  ✗ not found in {cid}: {repr(old[:50])}...')
        lines = src.split('\n')
        result = []
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                result.append(line + '\n')
            elif line:
                result.append(line)
        cell['source'] = result
        cell['outputs'] = []
        cell['execution_count'] = None

if changed == 2:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'\nSaved. {changed}/2 changes applied.')
else:
    print(f'\nWARNING: only {changed}/2 changes applied.')
