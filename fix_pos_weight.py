import json

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)

OLD = 'criterion_taste = nn.BCEWithLogitsLoss()'

NEW = (
    '# ── Class-imbalance correction via pos_weight ─────────────────────────────────\n'
    '# citrus_fruity=1%, floral=2.4%, mineral=1.7% are severely under-represented.\n'
    '# pos_weight[i] = neg_count / pos_count  (capped at 20 to prevent loss instability).\n'
    '_pos_counts  = taste_train.sum(axis=0).clip(min=1)          # (8,) float32\n'
    '_neg_counts  = len(taste_train) - _pos_counts\n'
    '_pw_raw      = _neg_counts / _pos_counts\n'
    '_pw_capped   = np.minimum(_pw_raw, 20.0)\n'
    'pos_weight_t = torch.tensor(_pw_capped, dtype=torch.float32, device=DEVICE)\n'
    'print(f"{\'Axis\':<16}  {\'pos%\':>5}  {\'pos_weight\':>10}")\n'
    'print("-" * 36)\n'
    'for ax, pw, pct in zip(TASTE_AXES, _pw_capped, _pos_counts / len(taste_train) * 100):\n'
    '    print(f"{ax:<16}  {pct:>5.1f}  {pw:>10.2f}")\n'
    'print()\n'
    '\n'
    'criterion_taste = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)'
)

changed = 0
for cell in nb['cells']:
    if cell.get('id') == '1e57c8ba':
        src = ''.join(cell.get('source', []))
        if OLD in src:
            new_src = src.replace(OLD, NEW)
            lines = new_src.split('\n')
            result = []
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    result.append(line + '\n')
                elif line:
                    result.append(line)
            cell['source'] = result
            cell['outputs'] = []
            cell['execution_count'] = None
            changed += 1
            print('  ✓ Patched cell 1e57c8ba')
        else:
            print('  OLD string not found – showing criterion lines:')
            for line in src.split('\n'):
                if 'criterion' in line or 'BCEWith' in line:
                    print('   ', repr(line))

if changed:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Notebook saved.')
else:
    print('No changes made.')
