import json, uuid

path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells before: {len(nb['cells'])}")
print(f"Last cell type: {nb['cells'][-1]['cell_type']}")
print(f"Last cell first line: {nb['cells'][-1]['source'][0][:60]!r}")

code_lines = [
    "# \u2500\u2500 11  Save all results to Google Drive \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "import pickle, glob, shutil\n",
    "\n",
    "# \u2500\u2500 1. Bundle everything produced so far \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "snapshot = {}\n",
    "\n",
    "# CNN Scratch\n",
    'snapshot["cnn_scratch"] = {\n',
    '    "test_acc":         globals().get("test_acc"),\n',
    '    "macro_f1":         globals().get("macro_f1"),\n',
    '    "history":          globals().get("history_scratch"),\n',
    '    "all_preds":        globals().get("all_preds"),\n',
    '    "all_labels":       globals().get("all_labels"),\n',
    '    "confusion_matrix": globals().get("sc_cm"),\n',
    '    "per_class_acc":    globals().get("sc_per_class_acc"),\n',
    "}\n",
    "\n",
    "# ResNet-50\n",
    'snapshot["resnet50"] = {\n',
    '    "test_acc":         globals().get("rn_test_acc"),\n',
    '    "macro_f1":         globals().get("rn_f1"),\n',
    '    "history":          globals().get("hist_rn"),\n',
    '    "all_preds":        globals().get("rn_preds"),\n',
    '    "all_labels":       globals().get("rn_labels"),\n',
    '    "confusion_matrix": globals().get("cm"),\n',
    '    "per_class_acc":    globals().get("per_class_acc"),\n',
    "}\n",
    "\n",
    "# Text preprocessing artefacts\n",
    'snapshot["text"] = {\n',
    '    "VOCAB":            globals().get("VOCAB"),\n',
    '    "VOCAB_SIZE":       globals().get("VOCAB_SIZE"),\n',
    '    "MAX_SEQ_LEN":      globals().get("MAX_SEQ_LEN"),\n',
    '    "embedding_matrix": globals().get("embedding_matrix"),\n',
    '    "GRAPE_CLASSES":    globals().get("GRAPE_CLASSES"),\n',
    '    "GRAPE_TO_IDX":     globals().get("GRAPE_TO_IDX"),\n',
    '    "CLASS_WEIGHTS":    globals().get("CLASS_WEIGHTS"),\n',
    "}\n",
    "\n",
    "# Split metadata\n",
    'snapshot["splits"] = {\n',
    '    "SEED":       SEED,\n',
    '    "train_size": globals().get("train_size"),\n',
    '    "test_size":  globals().get("test_size"),\n',
    "}\n",
    "\n",
    "# \u2500\u2500 2. Save locally \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    'snap_local = os.path.join(str(LOCAL_WEIGHTS), "results_snapshot.pkl")\n',
    'with open(snap_local, "wb") as f:\n',
    "    pickle.dump(snapshot, f)\n",
    "snap_size = os.path.getsize(snap_local) / 1e6\n",
    'print(f"\u2713 results_snapshot.pkl saved locally  ({snap_size:.1f} MB)")\n',
    "\n",
    "# \u2500\u2500 3. Copy to Drive (Colab only) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "if IN_COLAB:\n",
    '    snap_drive = os.path.join(WEIGHTS_DIR, "results_snapshot.pkl")\n',
    "    shutil.copy2(snap_local, snap_drive)\n",
    '    print(f"\u2713 Copied to Drive: {snap_drive}")\n',
    "\n",
    "# \u2500\u2500 4. Bulk-sync figures not yet on Drive \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "if IN_COLAB:\n",
    '    local_pngs = glob.glob(os.path.join(str(LOCAL_FIGURES), "*.png"))\n',
    "    synced, skipped = 0, 0\n",
    "    for src in sorted(local_pngs):\n",
    "        fname = os.path.basename(src)\n",
    "        dest  = os.path.join(FIGURES_DIR, fname)\n",
    "        if os.path.exists(dest) and os.path.getsize(dest) == os.path.getsize(src):\n",
    "            skipped += 1\n",
    "            continue\n",
    "        shutil.copy2(src, dest)\n",
    '        print(f"  \u2713 Figure synced: {fname}")\n',
    "        synced += 1\n",
    '    print(f"\\nFigures: {synced} synced, {skipped} already up-to-date.")\n',
    "else:\n",
    '    print("(Local mode \u2014 figures already on disk, no Drive sync needed)")\n',
    "\n",
    "# \u2500\u2500 5. Summary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "print(\"\\n\u2500\u2500 Snapshot contents \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\")\n",
    "for section, data in snapshot.items():\n",
    "    for key, val in data.items():\n",
    '        status = "\u2713" if val is not None else "\u2717  not in memory"\n',
    '        print(f"  {section:<14} {key:<22} {status}")\n',
    "\n",
    'print("\\n\u2713 Section 11 complete \u2014 all results saved.")\n',
    'print("  Reload in a future session with:")\n',
    'print("    import pickle")\n',
    "print(\"    snap = pickle.load(open('weights/results_snapshot.pkl', 'rb'))\")\n",
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": uuid.uuid4().hex[:8],
    "metadata": {},
    "outputs": [],
    "source": code_lines,
}

nb["cells"].append(new_cell)
print(f"Total cells after: {len(nb['cells'])}")
print(f"New cell id: {new_cell['id']}")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done. Saved successfully.")
