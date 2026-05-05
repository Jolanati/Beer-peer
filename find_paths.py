import json
nb = json.load(open(r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if ('BASE_DIR' in src or 'LOCAL_WEIGHTS' in src) and 'Path' in src and i < 10:
        print(i+1, c.get('id',''))
        print(src[:500])
        print('---')
