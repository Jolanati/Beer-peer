import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open(r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb', encoding='utf-8-sig'))
for cell in nb['cells']:
    if cell.get('id') == 'a6458717':
        with open('cell_145.txt', 'w', encoding='utf-8') as f:
            f.write(''.join(cell.get('source', [])))
        print('Found — written to cell_145.txt')
        break
