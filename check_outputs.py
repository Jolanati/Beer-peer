import json
path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
target_ids = ['VSC-df25a0d2', 'VSC-910390c2', 'VSC-4fd5b862']
for cell in cells:
    cid = cell.get('id','')
    if cid in target_ids:
        outputs = cell.get('outputs', [])
        print(f'--- {cid} ({len(outputs)} outputs) ---')
        for out in outputs:
            otype = out.get('output_type','')
            if otype == 'stream':
                text = ''.join(out.get('text', []))
                print(text[:600])
        print()
