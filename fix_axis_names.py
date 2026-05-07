import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)

OLD = '''\
TASTE_AXIS_NAMES = {
    "body":       "The Velvet",    # full, round, heavy, plush
    "acidity":    "The Electric",  # crisp, bright, zingy, sharp
    "tannin":     "The Grippy",    # firm, chewy, drying, bold
    "red_fruit":  "The Berry",     # cherry, raspberry, strawberry
    "dark_fruit": "The Dark",      # plum, blackberry, cassis, fig
    "earthy":     "The Earthy",    # leather, tobacco, soil, mushroom
    "sweet":      "The Honey",     # jammy, soft, warm, generous
    "oaky":       "The Smoky",     # vanilla, toast, cedar, smoke
    "floral":     "The Floral",    # violet, rose, jasmine, delicate
    "mineral":    "The Mineral",   # flinty, saline, chalky, stony
}'''

NEW = '''\
TASTE_AXIS_NAMES = {
    "body":       "rich",      # full, round, heavy, plush
    "acidity":    "crispy",    # crisp, bright, zingy, sharp
    "tannin":     "bold",      # firm, chewy, drying, grippy
    "red_fruit":  "juicy",     # cherry, raspberry, strawberry
    "dark_fruit": "deep",      # plum, blackberry, cassis, fig
    "earthy":     "earthy",    # leather, tobacco, soil, mushroom
    "sweet":      "sweet",     # jammy, soft, warm, generous
    "oaky":       "smoky",     # vanilla, toast, cedar, smoke
    "floral":     "delicate",  # violet, rose, jasmine, blossom
    "mineral":    "stony",     # flinty, saline, chalky, volcanic
}'''

changed = 0
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if OLD in src:
        new_src = src.replace(OLD, NEW)
        # Reconstruct as list of lines
        lines = new_src.split('\n')
        cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        cell['outputs'] = []
        cell['execution_count'] = None
        changed += 1
        print(f'  ✓ Updated TASTE_AXIS_NAMES in cell {cell.get("id")}')

if changed:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'Saved — {changed} cell(s) patched.')
else:
    print('WARNING: pattern not found — no changes made.')
