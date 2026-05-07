import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open(r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb', encoding='utf-8-sig'))
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'TASTE_AXIS_NAMES' in src and 'body' in src and 'acidity' in src and '=' in src:
        print('ID:', cell.get('id'))
        print(src[:800])
        print('---')
