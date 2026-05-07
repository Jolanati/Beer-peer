import json

with open(r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb', encoding='utf-8-sig') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'TASTE_SEEDS' in src:
        print('id:', cell.get('id'))
        print(src[:4000])
        print('---')
        break

# Also find threshold usage
print('\n=== threshold search ===')
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if '>= 0.5' in src and 'taste' in src.lower():
        print('threshold cell id:', cell.get('id'))
        print(src[:500])
        print('---')
