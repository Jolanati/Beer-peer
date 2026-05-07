import json

with open(r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb', encoding='utf-8-sig') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'TASTE_AXES' in src and 'citrus_fruity' in src and 'lemon' in src:
        print('=== TASTE_SEEDS cell id:', cell.get('id'), '===')
        print(src)
        break
