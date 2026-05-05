import json
nb = json.load(open(r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb', encoding='utf-8'))
cells = {c.get('id',''): c for c in nb['cells']}
for cid in ['sec13-1-code', 'sec13-4-code', 'sec13-5-code']:
    c = cells[cid]
    ec = c.get('execution_count')
    print(f'--- {cid} (exec_count={ec}) ---')
    for o in c.get('outputs', []):
        if o.get('output_type') == 'stream':
            txt = ''.join(o.get('text', []))
            print(txt[-700:])
    print()
