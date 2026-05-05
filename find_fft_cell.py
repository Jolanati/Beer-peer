import json
nb = json.load(open(r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'food_flavor_table.json' in src and 'WEIGHTS.parent' in src:
        cid = c.get('id', '')
        print(f'Cell {i+1}, id={cid}')
        print(src[:800])
        print()
