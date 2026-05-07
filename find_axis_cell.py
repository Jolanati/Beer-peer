import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open(r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb', encoding='utf-8-sig'))
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'TASTE_AXIS_NAMES' in src and '"The Velvet"' in src:
        print('ID:', cell.get('id'))
        with open('taste_axis_cell.txt', 'w', encoding='utf-8') as f:
            f.write(src)
        print(src)
