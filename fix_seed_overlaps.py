import json

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)

# Two targeted string replacements in cell 49bd6eb3
FIXES = [
    # Remove "tangy" from citrus_fruity — overlaps with acidic vocabulary
    (
        '                      "passion fruit","pineapple",   "tangy",       "yuzu",',
        '                      "passion fruit","pineapple",   "yuzu",',
    ),
    # Remove "iron" from mineral — matches "iron grip / iron-fisted tannins"
    (
        '                      "steely",   "iron",         "sea spray", "oyster shell"],',
        '                      "steely",   "sea spray",    "oyster shell"],',
    ),
]

changed = 0
for cell in nb['cells']:
    if cell.get('id') == '49bd6eb3':
        src = ''.join(cell.get('source', []))
        for old, new in FIXES:
            if old in src:
                src = src.replace(old, new)
                changed += 1
                print(f'  ✓ replaced: {repr(old[:60])}...')
            else:
                print(f'  ✗ not found: {repr(old[:60])}...')
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
    print(f'\nSaved ({changed} fixes applied).')
else:
    print(f'\nWARNING: only {changed}/2 fixes applied — notebook NOT saved.')
