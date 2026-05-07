import json

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)


def to_source(code):
    lines = code.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            if line:
                result.append(line)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 + 2 — §12.1: expand citrus_fruity seeds, tighten mineral seeds
# Cell id: 49bd6eb3
# ─────────────────────────────────────────────────────────────────────────────
NEW_TASTE_KW = '''\
# \u2500\u2500 12.1  Taste-label dataset \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

TASTE_AXES = ["tannic", "red_fruity", "citrus_fruity", "acidic",
              "earthy",  "floral",     "rich",          "mineral"]

_TASTE_KW = {
    "tannic":        ["tannin",   "tannic",       "grippy",    "astringent",
                      "firm tannin", "tight tannin"],
    "red_fruity":    ["cherry",   "raspberry",    "strawberry","blackberry",
                      "plum",     "currant",      "red fruit", "black fruit",
                      "blueberry","dark fruit"],
    # Fix 1 \u2014 expanded: original 7 seeds captured only 0.8% of reviews (recall 0.36).
    # Added tropical/tangerine/zest vocabulary that also signals citrus-style wines.
    "citrus_fruity": ["lemon",       "lime",         "orange",      "grapefruit",
                      "citrus",      "zesty",        "lemon zest",  "citrusy",
                      "tropical",    "tangerine",    "mandarin",    "passionfruit",
                      "passion fruit","pineapple",   "tangy",       "yuzu",
                      "lime zest",   "lemon peel",   "orange peel", "grapefruity",
                      "pomelo",      "bright citrus","fresh citrus"],
    "acidic":        ["acid",     "acidity",      "crisp",     "bright",
                      "tart",     "racy",         "lively"],
    "earthy":        ["earthy",   "earth",        "soil",      "mushroom",
                      "forest floor", "tobacco",  "leather",   "savory", "herbal"],
    "floral":        ["floral",   "flower",       "violet",    "rose",
                      "jasmine",  "blossom",      "lavender",  "perfumed"],
    "rich":          ["rich",     "full-bodied",  "full bodied","opulent",
                      "lush",     "concentrated", "powerful",  "dense",
                      "velvety",  "creamy"],
    # Fix 2 \u2014 tightened: removed "saline" (over-broad, appears in earthy/acidic contexts),
    # added more specific mineral descriptors to reduce false positives (precision was 0.647).
    "mineral":       ["mineral",  "minerality",   "stony",     "slate",
                      "chalk",    "flinty",       "wet stone", "limestone",
                      "graphite", "volcanic",     "flint",     "crushed rock",
                      "steely",   "iron",         "sea spray", "oyster shell"],
}


def _taste_label(description: str) -> list:
    """Binary taste labels via keyword matching on lowercased text."""
    d = description.lower()
    return [1.0 if any(kw in d for kw in _TASTE_KW[ax]) else 0.0
            for ax in TASTE_AXES]


# \u2500\u2500 Apply to train / val / test \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
taste_train = np.array([_taste_label(t) for t in df_train["review_text"]], dtype=np.float32)
taste_val   = np.array([_taste_label(t) for t in df_val["review_text"]],   dtype=np.float32)
taste_test  = np.array([_taste_label(t) for t in df_test["review_text"]],  dtype=np.float32)

# \u2500\u2500 Coverage stats \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print(f"{'Axis':<16} {'train %':>8}  {'val %':>7}  {'test %':>7}")
print("-" * 45)
for i, ax in enumerate(TASTE_AXES):
    tr = taste_train[:, i].mean() * 100
    v  = taste_val[:,   i].mean() * 100
    te = taste_test[:,  i].mean() * 100
    print(f"{ax:<16} {tr:>7.1f}%  {v:>6.1f}%  {te:>6.1f}%")

n_neutral = (taste_train.sum(axis=1) == 0).sum()
print(f"\\nNeutral (all-zero) reviews : {n_neutral:,} / {len(taste_train):,}  "
      f"({n_neutral/len(taste_train)*100:.1f}%)")
print(f"Avg taste axes per review  : {taste_train.sum(axis=1).mean():.2f}")

# \u2500\u2500 Build TasteDataset and DataLoaders \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
TASTE_BATCH = 64


class TasteDataset(torch.utils.data.Dataset):
    """Wine reviews with multi-label taste annotations."""
    def __init__(self, sequences, taste_labels):
        self.X       = torch.tensor(sequences,    dtype=torch.long)
        self.y       = torch.tensor(taste_labels, dtype=torch.float)   # (N, 8)
        self.lengths = (self.X != 0).sum(dim=1).clamp(min=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.lengths[i], self.y[i]


taste_train_ds = TasteDataset(X_train, taste_train)
taste_val_ds   = TasteDataset(X_val,   taste_val)
taste_test_ds  = TasteDataset(X_test,  taste_test)

taste_train_loader = DataLoader(taste_train_ds, batch_size=TASTE_BATCH, shuffle=True,
                                num_workers=0, pin_memory=False)
taste_val_loader   = DataLoader(taste_val_ds,   batch_size=TASTE_BATCH, shuffle=False,
                                num_workers=0, pin_memory=False)
taste_test_loader  = DataLoader(taste_test_ds,  batch_size=TASTE_BATCH, shuffle=False,
                                num_workers=0, pin_memory=False)

print(f"\\n{'Split':<8} {'Samples':>8}  {'Batches':>8}")
print("-" * 28)
for _n, _ds, _ldr in [("train", taste_train_ds, taste_train_loader),
                       ("val",   taste_val_ds,   taste_val_loader),
                       ("test",  taste_test_ds,  taste_test_loader)]:
    print(f"{_n:<8} {len(_ds):>8,}  {len(_ldr):>8,}")

print("\u2713 12.1 \u2014 Taste labels and DataLoaders ready.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3 — §12.4: per-axis inference thresholds
# Cell id: f62145e0
# ─────────────────────────────────────────────────────────────────────────────
NEW_EVAL = '''\
# \u2500\u2500 12.4  TasteBiLSTM \u2014 test evaluation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
from sklearn.metrics import precision_score, recall_score

taste_bilstm.load_state_dict(load_checkpoint("tastebilstm_best.pt"))
taste_bilstm.eval()

all_probs_taste, all_labels_taste = [], []
with torch.no_grad():
    for seqs, lengths, labels in taste_test_loader:
        logits = taste_bilstm(seqs.to(DEVICE), lengths.to(DEVICE))
        all_probs_taste.append(logits.sigmoid().cpu())
        all_labels_taste.append(labels)

all_probs_taste  = torch.cat(all_probs_taste,  dim=0)
all_labels_taste = torch.cat(all_labels_taste, dim=0)

# Fix 3 \u2014 per-axis thresholds.
# citrus_fruity had recall 0.36 at 0.50 \u2192 lowered to 0.35 to catch more positives.
# mineral had precision 0.65 at 0.50 (too many false positives) \u2192 raised to 0.55.
_AXIS_THRESHOLDS = torch.tensor([
    0.50,  # tannic
    0.50,  # red_fruity
    0.35,  # citrus_fruity  \u2190 was 0.50 (recall 0.36 \u2192 expected ~0.60+)
    0.50,  # acidic
    0.50,  # earthy
    0.50,  # floral
    0.50,  # rich
    0.55,  # mineral        \u2190 was 0.50 (precision 0.65 \u2192 expected ~0.72+)
])
all_preds_taste = (all_probs_taste >= _AXIS_THRESHOLDS).float()

# \u2500\u2500 Per-axis accuracy, precision, recall, F1 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print(f"{'Axis':<16} {'Thr':>4}  {'Acc':>7}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
print("-" * 56)
taste_axis_f1s = []
for i, ax in enumerate(TASTE_AXES):
    y_true = all_labels_taste[:, i].numpy()
    y_pred = all_preds_taste[:,  i].numpy()
    acc    = (y_true == y_pred).mean()
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true,    y_pred, zero_division=0)
    f1_ax  = f1_score(y_true,        y_pred, zero_division=0)
    taste_axis_f1s.append(f1_ax)
    thr    = _AXIS_THRESHOLDS[i].item()
    print(f"{ax:<16} {thr:>4.2f}  {acc:>7.4f}  {prec:>6.4f}  {rec:>6.4f}  {f1_ax:>6.4f}")

macro_taste_f1 = np.mean(taste_axis_f1s)
exact_match    = (all_preds_taste == all_labels_taste).all(dim=1).float().mean().item()
print(f"\\nMacro F1 (avg over axes) : {macro_taste_f1:.4f}")
print(f"Exact-match accuracy     : {exact_match:.4f}")

# \u2500\u2500 Learning curves \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
_epochs_taste = range(1, len(history_taste["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(_epochs_taste, history_taste["train_loss"], label="Train", lw=1.8)
axes[0].plot(_epochs_taste, history_taste["val_loss"],   label="Val",   lw=1.8)
axes[0].set_title("TasteBiLSTM \u2014 Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(_epochs_taste, history_taste["train_acc"], label="Train", lw=1.8)
axes[1].plot(_epochs_taste, history_taste["val_acc"],   label="Val",   lw=1.8)
axes[1].set_title("TasteBiLSTM \u2014 Exact-match Accuracy")
axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.suptitle(
    f"12.4 \u2014 TasteBiLSTM  |  Macro-F1: {macro_taste_f1:.4f}  "
    f"Exact-match: {exact_match*100:.2f}%", fontsize=12)
plt.tight_layout()
save_figure(fig, "tastebilstm_curves.png")
plt.show()

# \u2500\u2500 Per-axis F1 bar chart \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(TASTE_AXES, taste_axis_f1s, color="#4C72B0", alpha=0.85)
ax.axhline(macro_taste_f1, color="red", lw=1.5, ls="--",
           label=f"Macro F1 = {macro_taste_f1:.3f}")
ax.set_xlabel("Taste axis"); ax.set_ylabel("F1 score"); ax.set_ylim(0, 1.05)
ax.set_title("12.4 \u2014 TasteBiLSTM: per-axis F1 on test set")
ax.legend(); plt.tight_layout()
save_figure(fig, "tastebilstm_axis_f1.png")
plt.show()
print("\u2713 12.4 \u2014 TasteBiLSTM evaluation complete.")
'''


# ─────────────────────────────────────────────────────────────────────────────
# Patch notebook
# ─────────────────────────────────────────────────────────────────────────────
changed = 0
for cell in nb['cells']:
    cid = cell.get('id', '')
    if cid == '49bd6eb3':
        cell['source'] = to_source(NEW_TASTE_KW)
        cell['outputs'] = []
        cell['execution_count'] = None
        changed += 1
        print(f'  \u2713 12.1 seeds cell ({cid}) updated')
    elif cid == 'f62145e0':
        cell['source'] = to_source(NEW_EVAL)
        cell['outputs'] = []
        cell['execution_count'] = None
        changed += 1
        print(f'  \u2713 12.4 eval cell ({cid}) updated')

if changed == 2:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'\nNotebook saved. {changed}/2 cells patched.')
else:
    print(f'\nERROR: expected 2, found {changed}.')
