import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open(r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb', encoding='utf-8-sig'))
for cell in nb['cells']:
    if cell.get('id') == '1860746a':
        print(''.join(cell.get('source', [])))
