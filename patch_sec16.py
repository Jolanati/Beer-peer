import json, uuid

NB_PATH = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

def new_cell(cell_type, source):
    c = {"cell_type": cell_type, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source}
    if cell_type == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c

# find insertion point: after §15.8 code cell
insert_after = None
for i, c in enumerate(cells):
    if 'joint_random_predictor.png' in ''.join(c.get('source', [])) and c['cell_type'] == 'code':
        insert_after = i
        break
print(f"Inserting §16 after cell #{insert_after + 1}")

# ── §16 markdown ──────────────────────────────────────────────────────────────
SEC16_MD = """\
## 16 — Deployment Prototype

An interactive **Gradio** app that demonstrates the full Wine & Dine pipeline in three steps:

1. **Upload a food photo** — any food photo, real or from the dataset.
2. **CNN identifies the food** — ResNet-50 predicts the food category with a confidence score and top-5 alternatives shown as a bar chart. The app asks *"Is this right?"*
3. **Confirm → wine pairing card** — the BiLSTM pipeline looks up the food's flavor profile and returns three personalised wine recommendations: *Safe Bet*, *Characteristic*, and *Contrast*.

The app runs directly in this Colab cell.  `share=True` produces a temporary public URL (valid 72 h) — copy it to share with the class or include in the submission.

> **Screenshot:** run the cell, open the Gradio link, upload a food photo, and take a screenshot to satisfy the §5.7 deliverable.
"""

# ── §16 install cell ──────────────────────────────────────────────────────────
SEC16_INSTALL = """\
# ── Install Gradio if not already present ─────────────────────────────────────
try:
    import gradio as gr
    print(f"gradio {gr.__version__} already installed.")
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
    import gradio as gr
    print(f"Installed gradio {gr.__version__}.")
"""

# ── §16 Gradio app ────────────────────────────────────────────────────────────
SEC16_APP = r"""# ── 16  Wine & Dine — Gradio Deployment ──────────────────────────────────────
import gradio as gr
import torch
from PIL import Image

# ── Inference helpers ─────────────────────────────────────────────────────────
def _cnn_identify(pil_img):
    """ResNet-50 → (food_name, confidence, top5 list of (name, prob))"""
    resnet50.eval()
    img_t = val_test_transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = resnet50(img_t)
        probs  = torch.softmax(logits, 1)
        top5_p, top5_i = probs.topk(5, dim=1)
    top5 = [
        (_food101_classes[int(top5_i[0, k])].replace("_", " ").title(),
         float(top5_p[0, k]))
        for k in range(5)
    ]
    return top5[0][0], top5[0][1], top5

def _wine_card_html(food_name, cnn_conf, top5):
    """Build HTML wine pairing card for the confirmed food."""
    food_key     = food_name.lower().replace(" ", "_")
    cluster_idx  = food_to_cluster.get(food_key, -1)
    cluster_name = cluster_names.get(cluster_idx, "—") if cluster_idx >= 0 else "—"
    recs         = results_all.get(food_key, [])

    TIER_LABELS = ["🥇 Safe Bet", "✨ Characteristic", "🔄 Contrast"]
    wine_rows = ""
    for i, rec in enumerate(recs[:3]):
        label    = TIER_LABELS[i] if i < len(TIER_LABELS) else f"Option {i+1}"
        wine     = rec.get("wine", "—")
        rating   = rec.get("rating_pt", "—")
        raw_desc = rec.get("description", "")
        desc     = (raw_desc[:110] + "…") if len(raw_desc) > 110 else (raw_desc or "—")
        wine_rows += f"""
        <tr>
          <td style="padding:8px 14px;font-weight:bold;color:#8B0000;white-space:nowrap">{label}</td>
          <td style="padding:8px 14px;font-weight:600;color:#1a1a2e">{wine}</td>
          <td style="padding:8px 6px;color:#666;font-size:12px;white-space:nowrap">{rating} pts</td>
          <td style="padding:8px 14px;color:#777;font-size:12px;font-style:italic">{desc}</td>
        </tr>"""

    if not wine_rows:
        wine_rows = '<tr><td colspan="4" style="padding:14px;color:#aaa;text-align:center">No wine data available for this food.</td></tr>'

    # top-5 horizontal bars
    top5_bars = ""
    for fn, fp in top5:
        bar_w   = int(fp * 220)
        col     = "#2CA02C" if fn == food_name else "#d0ccc8"
        weight  = "700" if fn == food_name else "400"
        top5_bars += f"""
        <div style="display:flex;align-items:center;margin:4px 0;font-size:12px">
          <span style="width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                       color:#444;font-weight:{weight}">{fn}</span>
          <div style="background:{col};width:{bar_w}px;height:13px;
                      border-radius:3px;margin:0 8px;transition:width 0.3s"></div>
          <span style="color:#888;font-weight:{weight}">{fp*100:.0f}%</span>
        </div>"""

    return f"""
<div style="font-family:'Segoe UI',sans-serif;background:#faf7f2;border-radius:14px;
            padding:28px 32px;max-width:700px;box-shadow:0 3px 16px rgba(0,0,0,0.10)">

  <div style="font-size:24px;font-weight:800;color:#1a1a2e;margin-bottom:4px">
    🍽️&nbsp; {food_name}
  </div>
  <div style="font-size:13px;color:#999;margin-bottom:20px">
    ResNet-50 confidence:&nbsp;<strong style="color:#2CA02C">{cnn_conf*100:.0f}%</strong>
    &nbsp;·&nbsp; Flavor cluster:&nbsp;<strong style="color:#6b3fa0">{cluster_name}</strong>
  </div>

  <div style="font-size:11px;color:#bbb;text-transform:uppercase;
              letter-spacing:1.2px;margin-bottom:8px">Top-5 predictions</div>
  {top5_bars}

  <hr style="margin:22px 0;border:none;border-top:1px solid #e4ddd2">

  <div style="font-size:11px;color:#bbb;text-transform:uppercase;
              letter-spacing:1.2px;margin-bottom:12px">Wine Pairings</div>
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr style="background:#f0ebe0;font-size:11px;color:#999;text-transform:uppercase">
        <th style="padding:7px 14px;text-align:left">Tier</th>
        <th style="padding:7px 14px;text-align:left">Wine</th>
        <th style="padding:7px 6px;text-align:left">Rating</th>
        <th style="padding:7px 14px;text-align:left">Tasting note</th>
      </tr>
    </thead>
    <tbody>{wine_rows}</tbody>
  </table>

  <div style="margin-top:20px;font-size:11px;color:#ccc;text-align:right">
    Wine &amp; Dine · RSU Advanced ML · 2026
  </div>
</div>"""

# ── App state (simple dict — Gradio Blocks share closure state) ───────────────
_state = {"food": "", "conf": 0.0, "top5": []}

def on_identify(pil_img):
    if pil_img is None:
        return (
            "*Please upload a food photo first.*",
            gr.update(visible=False),
            gr.update(visible=False),
            ""
        )
    food_name, conf, top5 = _cnn_identify(pil_img)
    _state["food"] = food_name
    _state["conf"] = conf
    _state["top5"] = top5
    runner_up = f"{top5[1][0]}  ({top5[1][1]*100:.0f}%)" if len(top5) > 1 else "—"
    msg = f"""## 🔍 I think this is…

# {food_name}

I'm **{conf*100:.0f}% confident** in that.
*(Runner-up: {runner_up})*

---
**Is that right?**"""
    return msg, gr.update(visible=True), gr.update(visible=False), ""

def on_yes():
    html = _wine_card_html(_state["food"], _state["conf"], _state["top5"])
    return gr.update(visible=True), html

def on_no():
    return (
        "*Upload another photo and click **Identify Food** to try again.*",
        gr.update(visible=False),
        gr.update(visible=False),
        ""
    )

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="orange"),
    title="Wine & Dine 🍷"
) as demo:

    gr.Markdown("""# 🍽️ Wine & Dine
    **Upload a food photo — we identify it and find your perfect wine pairing.**
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            img_input    = gr.Image(type="pil", label="📷 Food photo", height=340)
            identify_btn = gr.Button("🔍 Identify Food", variant="primary", size="lg")

        with gr.Column(scale=1, min_width=320):
            prediction_md = gr.Markdown("*Upload a photo and click **Identify Food** to begin.*")

            with gr.Row(visible=False) as confirm_row:
                yes_btn = gr.Button("✅  Yes — show me wine pairings!", variant="primary")
                no_btn  = gr.Button("❌  No, try another photo",        variant="secondary")

    with gr.Group(visible=False) as card_group:
        gr.Markdown("---\n## 🍷 Your Wine Pairings")
        wine_card = gr.HTML()

    # ── Event wiring ──────────────────────────────────────────────────────────
    identify_btn.click(
        on_identify,
        inputs=[img_input],
        outputs=[prediction_md, confirm_row, card_group, wine_card]
    )
    yes_btn.click(
        on_yes,
        inputs=None,
        outputs=[card_group, wine_card]
    )
    no_btn.click(
        on_no,
        inputs=None,
        outputs=[prediction_md, confirm_row, card_group, wine_card]
    )

print("Launching Wine & Dine…  (share link valid for 72 h)")
demo.launch(share=True, debug=False, quiet=True)
"""

new_cells = [
    new_cell("markdown", SEC16_MD),
    new_cell("code",     SEC16_INSTALL),
    new_cell("code",     SEC16_APP),
]

for j, nc in enumerate(new_cells):
    cells.insert(insert_after + 1 + j, nc)
    print(f"  Inserted '{nc['cell_type']}' cell #{insert_after + 2 + j}")

nb['cells'] = cells
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✓ §16 written.")
