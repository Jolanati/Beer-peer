import json

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)

FIXES = {
    # ── §12.1: add oaky + sweet axes ─────────────────────────────────────────
    '49bd6eb3': [
        (
            'TASTE_AXES = ["tannic", "red_fruity", "citrus_fruity", "acidic",\n              "earthy",  "floral",     "rich",          "mineral"]',
            'TASTE_AXES = ["tannic", "red_fruity", "citrus_fruity", "acidic",\n              "earthy",  "floral",     "rich",          "mineral",\n              "oaky",    "sweet"]',
        ),
        (
            '    # Fix 2 — tightened: removed "saline" (over-broad, appears in earthy/acidic contexts),\n    # added more specific mineral descriptors to reduce false positives (precision was 0.647).\n    "mineral":       ["mineral",  "minerality",   "stony",     "slate",\n                      "chalk",    "flinty",       "wet stone", "limestone",\n                      "graphite", "volcanic",     "flint",     "crushed rock",\n                      "steely",   "sea spray",    "oyster shell"],\n}',
            '    # Fix 2 — tightened: removed "saline" (over-broad, appears in earthy/acidic contexts),\n    # added more specific mineral descriptors to reduce false positives (precision was 0.647).\n    "mineral":       ["mineral",  "minerality",   "stony",     "slate",\n                      "chalk",    "flinty",       "wet stone", "limestone",\n                      "graphite", "volcanic",     "flint",     "crushed rock",\n                      "steely",   "sea spray",    "oyster shell"],\n    # New axis: oak aging — major pairing divider (oaked vs unoaked changes everything)\n    "oaky":          ["oak",      "oaky",         "vanilla",   "toast",\n                      "toasty",   "cedar",        "smoky",     "barrel",\n                      "wood",     "woody",        "buttery",   "spice",\n                      "clove",    "coconut",      "caramel"],\n    # New axis: sweetness — critical for spicy food, dessert, and cheese pairings\n    "sweet":         ["sweet",    "sweetness",    "off-dry",   "residual sugar",\n                      "honeyed",  "honey",        "dessert",   "luscious",\n                      "syrupy",   "jammy",        "ripe",      "candied",\n                      "sugar",    "semi-sweet"],\n}',
        ),
    ],
    # ── §12.4: add oaky + sweet thresholds ───────────────────────────────────
    'f62145e0': [
        (
            '_AXIS_THRESHOLDS = torch.tensor([\n    0.50,  # tannic\n    0.50,  # red_fruity\n    0.35,  # citrus_fruity  ← was 0.50 (recall 0.36 → expected ~0.60+)\n    0.50,  # acidic\n    0.50,  # earthy\n    0.50,  # floral\n    0.50,  # rich\n    0.55,  # mineral        ← was 0.50 (precision 0.65 → expected ~0.72+)\n])',
            '_AXIS_THRESHOLDS = torch.tensor([\n    0.50,  # tannic\n    0.50,  # red_fruity\n    0.35,  # citrus_fruity  ← lowered: recall was 0.36\n    0.50,  # acidic\n    0.50,  # earthy\n    0.50,  # floral\n    0.50,  # rich\n    0.55,  # mineral        ← raised: precision was 0.65\n    0.50,  # oaky           new axis\n    0.50,  # sweet          new axis\n])',
        ),
    ],
    # ── §14.2: K 8 → 10 ──────────────────────────────────────────────────────
    '8839e43f': [
        (
            'K_CLUSTERS = 8   # 8 matches the 8 TasteBiLSTM axes; 12 over-splits into duplicate clusters',
            'K_CLUSTERS = 10  # 10 matches the 10 TasteBiLSTM axes; BisectingKMeans makes 9 splits',
        ),
    ],
}

total_expected = sum(len(v) for v in FIXES.values())
changed = 0

for cell in nb['cells']:
    cid = cell.get('id', '')
    if cid not in FIXES:
        continue
    src = ''.join(cell.get('source', []))
    for old, new in FIXES[cid]:
        if old in src:
            src = src.replace(old, new)
            changed += 1
            print(f'  ✓ {cid}: {repr(old[:55])}...')
        else:
            print(f'  ✗ NOT FOUND {cid}: {repr(old[:55])}...')
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

if changed == total_expected:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'\nSaved. {changed}/{total_expected} changes applied.')
else:
    print(f'\nWARNING: only {changed}/{total_expected} applied — NOT saved.')
