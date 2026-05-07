"""
Wine & Dine — Gradio Deployment App
HuggingFace Spaces: https://huggingface.co/spaces/Jolanati/wine-dine

Pipeline (all inference happens in real time):
  1. Upload food photo
  2. ResNet-50 → food class + confidence + top-5
  3. User confirms → "Yes, that's my dish"
  4. Flavor description loaded from table
  5. BiLSTM encodes description live → 512-d taste vector
  6. Cosine similarity to saved cluster centroids → flavor cluster
  7. Wine card: Safe Bet / Characteristic / Contrast
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models
import torchvision.transforms as T
import gradio as gr
from PIL import Image

# ── Paths (relative to app.py — HF Spaces serves from repo root) ─────────────
BASE_DIR    = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DATA_DIR    = os.path.join(BASE_DIR, "data")

CNN_WEIGHTS    = os.path.join(WEIGHTS_DIR, "cnn_resnet50_best.pt")
BILSTM_WEIGHTS = os.path.join(WEIGHTS_DIR, "bilstm_best.pt")
DATA_JSON      = os.path.join(DATA_DIR,    "food_flavor_description_v2.json")
VOCAB_JSON     = os.path.join(DATA_DIR,    "vocab.json")
CLUSTER_JSON   = os.path.join(DATA_DIR,    "cluster_names.json")
RESULTS_JSON   = os.path.join(DATA_DIR,    "results_all.json")
CENTROIDS_NPY  = os.path.join(DATA_DIR,    "centroids.npy")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── BiLSTM hyperparameters (must match training) ──────────────────────────────
HIDDEN_DIM  = 256
N_LAYERS    = 2
DROPOUT_RNN = 0.4
EMBED_DIM   = 100
MAX_SEQ_LEN = 64
GRAPE_CLASSES = [
    "Bordeaux-style Red Blend","Cabernet Sauvignon","Chardonnay",
    "Merlot","Pinot Gris","Pinot Noir","Red Blend","Riesling",
    "Rosé","Sauvignon Blanc","Sparkling Blend","Syrah","White Blend",
    "Zinfandel","Other",
]

# ── Food-101 class list (alphabetical — matches torchvision ImageFolder order) ─
FOOD101_CLASSES = [
    "apple_pie","baby_back_ribs","baklava","beef_carpaccio","beef_tartare",
    "beet_salad","beignets","bibimbap","bread_pudding","breakfast_burrito",
    "bruschetta","caesar_salad","cannoli","caprese_salad","carrot_cake",
    "ceviche","cheese_plate","cheesecake","chicken_curry","chicken_quesadilla",
    "chicken_wings","chocolate_cake","chocolate_mousse","churros","clam_chowder",
    "club_sandwich","crab_cakes","creme_brulee","croque_madame","cup_cakes",
    "deviled_eggs","donuts","dumplings","edamame","eggs_benedict","escargots",
    "falafel","filet_mignon","fish_and_chips","foie_gras","french_fries",
    "french_onion_soup","french_toast","fried_calamari","fried_rice",
    "frozen_yogurt","garlic_bread","gnocchi","greek_salad","grilled_cheese_sandwich",
    "grilled_salmon","guacamole","gyoza","hamburger","hot_and_sour_soup","hot_dog",
    "huevos_rancheros","hummus","ice_cream","lasagna","lobster_bisque",
    "lobster_roll_sandwich","macaroni_and_cheese","macarons","miso_soup","mussels",
    "nachos","omelette","onion_rings","oysters","pad_thai","paella","pancakes",
    "panna_cotta","peking_duck","pho","pizza","pork_chop","poutine","prime_rib",
    "pulled_pork_sandwich","ramen","ravioli","red_velvet_cake","risotto","samosa",
    "sashimi","scallops","seaweed_salad","shrimp_and_grits","spaghetti_bolognese",
    "spaghetti_carbonara","spring_rolls","steak","strawberry_shortcake","sushi",
    "tacos","takoyaki","tiramisu","tuna_tartare","waffles",
]

# ── Image transform (ImageNet normalisation — same as training) ───────────────
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ── Load ResNet-50 ─────────────────────────────────────────────────────────────
def _load_resnet():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 101)
    if os.path.exists(CNN_WEIGHTS):
        state = torch.load(CNN_WEIGHTS, map_location=DEVICE)
        # handle nested checkpoint dict
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    else:
        print(f"WARNING: {CNN_WEIGHTS} not found — using random weights.")
    model.to(DEVICE).eval()
    return model

# ── Load flavor pairing data ──────────────────────────────────────────────────
def _load_flavor_data():
    if os.path.exists(DATA_JSON):
        with open(DATA_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}

resnet50     = _load_resnet()
flavor_data  = _load_flavor_data()

# ── Load BiLSTM artifacts ─────────────────────────────────────────────────────
def _safe_load(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    print(f"WARNING: {path} not found.")
    return default

VOCAB         = _safe_load(VOCAB_JSON, {})
CLUSTER_NAMES = {int(k): v for k, v in _safe_load(CLUSTER_JSON, {}).items()}
RESULTS_ALL   = _safe_load(RESULTS_JSON, {})
CENTROIDS     = np.load(CENTROIDS_NPY) if os.path.exists(CENTROIDS_NPY) else None
VOCAB_SIZE    = len(VOCAB) + 1

# ── BiLSTM architecture ───────────────────────────────────────────────────────
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states, mask=None):
        energy  = torch.tanh(self.attn(hidden_states))
        scores  = self.v(energy).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return (weights.unsqueeze(-1) * hidden_states).sum(dim=1), weights


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes,
                 n_layers=2, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                                 batch_first=True, bidirectional=True,
                                 dropout=dropout if n_layers > 1 else 0.0)
        self.attention = BahdanauAttention(hidden_dim)
        self.drop      = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, n_classes)

    def encode(self, x, lengths):
        emb    = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        total_length=x.shape[1])
        context, attn_w = self.attention(output, (x != 0))
        return context, attn_w   # (B, 512), (B, seq)


def _load_bilstm():
    model = BiLSTMAttention(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM,
                            len(GRAPE_CLASSES), N_LAYERS, DROPOUT_RNN)
    if os.path.exists(BILSTM_WEIGHTS):
        ckpt = torch.load(BILSTM_WEIGHTS, map_location=DEVICE)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        model.load_state_dict(ckpt, strict=False)
    else:
        print(f"WARNING: {BILSTM_WEIGHTS} not found — using random weights.")
    return model.to(DEVICE).eval()

bilstm = _load_bilstm()


def _tokenize(text):
    tokens = [VOCAB.get(w.lower(), 0) for w in str(text).split()]
    tokens = tokens[:MAX_SEQ_LEN]
    tokens += [0] * (MAX_SEQ_LEN - len(tokens))
    return tokens


def bilstm_encode(food_key):
    """Run BiLSTM on food's flavor description. Returns cluster info + attention."""
    entry = flavor_data.get(food_key, {})
    desc  = entry.get("classic", "balanced complex food")
    if isinstance(desc, list):
        desc = " ".join(desc)

    tokens  = _tokenize(desc)
    tok_t   = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    lengths = (tok_t != 0).sum(1).clamp(min=1)

    with torch.no_grad():
        vec, attn_w = bilstm.encode(tok_t, lengths)   # (1,512), (1,seq)

    vec_np  = vec.squeeze(0).cpu().numpy()
    attn_np = attn_w.squeeze(0).cpu().numpy()

    if CENTROIDS is not None:
        vec_l2  = vec_np / (np.linalg.norm(vec_np) + 1e-8)
        sims    = CENTROIDS @ vec_l2
        cluster = int(np.argmax(sims))
    else:
        sims, cluster = np.zeros(9), 0

    cluster_name = CLUSTER_NAMES.get(cluster, f"Cluster {cluster}")
    return cluster, cluster_name, sims, desc, attn_np

# ── Inference ─────────────────────────────────────────────────────────────────
def identify_food(pil_img):
    """Return (food_name, confidence_float, top5_list)."""
    img_t = TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = resnet50(img_t)
        probs  = torch.softmax(logits, 1)
        top5_p, top5_i = probs.topk(5, dim=1)
    top5 = [
        (FOOD101_CLASSES[int(top5_i[0, k])].replace("_", " ").title(),
         float(top5_p[0, k]))
        for k in range(5)
    ]
    return top5[0][0], top5[0][1], top5

# ── Card style helpers (matching notebook §14.5 print_card) ───────────────────
_AXIS_FOOD_FEEL = {
    "soft":     "rich and velvety",
    "crispy":   "bright and zesty",
    "bold":     "hearty and bold",
    "juicy":    "fruity and fresh",
    "deep":     "deep and ripe",
    "earthy":   "savory and earthy",
    "sweet":    "sweet and indulgent",
    "smoky":    "warm and smoky",
    "delicate": "light and fragrant",
    "mineral":  "crisp and mineral",
}

_INTENT = {
    "SAFE BET":   "matches it",
    "HIDDEN GEM": "surprises you",
    "BOLD MOVE":  "goes against it",
}

_TIER_COLOR = {
    "SAFE BET":   "#2CA02C",
    "HIDDEN GEM": "#1F77B4",
    "BOLD MOVE":  "#D62728",
}

_TIER_BORDER = {
    "SAFE BET":   "#2CA02C",
    "HIDDEN GEM": "#1F77B4",
    "BOLD MOVE":  "#D62728",
}


def _cluster_adj(cluster_name: str) -> str:
    first = cluster_name.split("&")[0].strip()
    if first.lower().startswith("the "):
        first = first[4:]
    if first.lower().startswith("something "):
        first = first[10:]
    return first.lower()


def _food_feel(safe_bet_cluster_name: str) -> str:
    adj = _cluster_adj(safe_bet_cluster_name)
    return _AXIS_FOOD_FEEL.get(adj, "rich and complex")


def _clip(text: str, max_chars: int = 160) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;: ") + "…"


def _conf_bar_html(conf: float, color: str) -> str:
    pct = int(conf * 100)
    bar_w = int(conf * 120)
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
        f'<div style="background:#e8e4dd;border-radius:4px;width:120px;height:10px;overflow:hidden">'
        f'<div style="background:{color};width:{bar_w}px;height:10px;border-radius:4px"></div>'
        f'</div>'
        f'<span style="font-size:12px;color:#888;font-weight:600">{pct}%</span>'
        f'</div>'
    )


# ── HTML builders ─────────────────────────────────────────────────────────────
def _top5_bars_html(top5, confirmed_food):
    html = ""
    for fn, fp in top5:
        w   = int(fp * 250)
        col = "#2CA02C" if fn == confirmed_food else "#d4cfc9"
        fw  = "700"    if fn == confirmed_food else "400"
        html += (
            f'<div style="display:flex;align-items:center;margin:3px 0;font-size:12px">'
            f'<span style="width:200px;overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap;color:#444;font-weight:{fw}">{fn}</span>'
            f'<div style="background:{col};width:{w}px;height:12px;'
            f'border-radius:3px;margin:0 8px"></div>'
            f'<span style="color:#888;font-weight:{fw}">{fp*100:.0f}%</span>'
            f'</div>'
        )
    return html


def _tier_card_html(rec: dict, display_name: str, feel: str) -> str:
    """Build one tier panel — matches notebook print_card style."""
    tier    = rec.get("tier", "")
    icon    = rec.get("icon", "")
    name    = rec.get("name", "")
    wine    = rec.get("wine", "—")
    rating  = rec.get("rating", "—")
    snippet = _clip(rec.get("snippet", ""))
    conf    = rec.get("confidence", 0)
    kws     = rec.get("keywords", [])

    color   = _TIER_COLOR.get(tier, "#555")
    border  = _TIER_BORDER.get(tier, "#ccc")
    intent  = _INTENT.get(tier, "pairs with")
    adj     = _cluster_adj(name)
    kw_str  = " · ".join(kws) if kws else "—"

    return f"""
    <div style="border-left:4px solid {border};background:#fff;
                border-radius:0 12px 12px 0;padding:18px 22px;margin:10px 0;
                box-shadow:0 2px 8px rgba(0,0,0,0.04)">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span style="font-size:18px">{icon}</span>
        <span style="font-size:15px;font-weight:700;color:{color}">{tier}</span>
      </div>
      <div style="font-size:11px;color:#aaa;text-transform:uppercase;
                  letter-spacing:0.8px;margin-bottom:4px">Match confidence</div>
      {_conf_bar_html(conf, color)}
      <div style="font-size:13px;color:#444;margin:12px 0 6px;line-height:1.6">
        As your <strong>{display_name}</strong> is <em>{feel}</em>,<br>
        we believe you need a wine that
        <strong style="color:{color}">{intent}</strong> —
        something <em style="font-weight:600">{adj}</em><br>
        that plays on
        <span style="color:#0D7C66;font-weight:600">{kw_str}</span>
      </div>
      <hr style="margin:12px 0;border:none;border-top:1px solid #eee">
      <div style="font-size:11px;color:#aaa;margin-bottom:4px">Wine drinkers suggest:</div>
      <div style="font-size:14px;font-weight:700;color:#1a1a2e">
        {wine} &nbsp;⭐ {rating}
      </div>
      <div style="font-size:11px;color:#aaa;margin:8px 0 2px">Wine drinkers say:</div>
      <div style="font-size:12px;color:#666;font-style:italic;
                  line-height:1.5;padding:6px 0">
        "{snippet}"
      </div>
    </div>"""


def _wine_card_html(food_name: str, conf: float, top5,
                    cluster_idx: int, cluster_name: str,
                    sims, desc: str, attn_w) -> str:
    """Build the full card (non-streaming fallback)."""
    parts = list(_wine_card_parts(food_name, conf, top5,
                                  cluster_idx, cluster_name,
                                  sims, desc, attn_w))
    return parts[-1] if parts else ""


def _wine_card_parts(food_name: str, conf: float, top5,
                     cluster_idx: int, cluster_name: str,
                     sims, desc: str, attn_w):
    """Yield progressively longer HTML — each yield is the full card so far."""
    food_key = food_name.lower().replace(" ", "_")
    recs     = RESULTS_ALL.get(food_key, [])
    display  = food_name.replace("_", " ").title() if "_" in food_name else food_name

    WRAP_OPEN = (
        '<div style="font-family:\'Segoe UI\',Arial,sans-serif;background:#faf7f2;'
        'border-radius:16px;padding:28px 34px;'
        'box-shadow:0 4px 20px rgba(0,0,0,0.09)">'
    )
    FOOTER = (
        '<div style="margin-top:18px;font-size:10px;color:#ccc;text-align:right">'
        'Wine &amp; Dine · RSU Advanced ML · 2026</div>'
    )

    # Header (always shown)
    header = f"""
  <div style="font-size:24px;font-weight:800;color:#1a1a2e;margin-bottom:2px">
    🍽️&nbsp; {display}
    <span style="font-size:12px;font-weight:400;color:#aaa;margin-left:10px">
      CNN confidence: <strong style="color:#2CA02C">{conf*100:.0f}%</strong>
    </span>
  </div>
  <details style="margin:12px 0 4px">
    <summary style="cursor:pointer;font-size:11px;color:#bbb;
                    text-transform:uppercase;letter-spacing:1px">
      📷 Top-5 CNN predictions
    </summary>
    <div style="margin-top:8px">{_top5_bars_html(top5, food_name)}</div>
  </details>
  <hr style="margin:18px 0;border:none;border-top:1px solid #e4ddd2">
"""

    # ── Stage 0: "Computing..." spinner ──────────────────────────────────────
    spinner = (
        '<div style="padding:20px;text-align:center;color:#6b3fa0;font-size:14px">'
        '⏳ Encoding flavor description through BiLSTM…</div>'
    )
    yield WRAP_OPEN + header + spinner + FOOTER + '</div>'

    # Attention-highlighted flavor text
    words    = desc.split()[:MAX_SEQ_LEN]
    attn_arr = attn_w[:len(words)]
    a_min, a_max = attn_arr.min(), attn_arr.max()
    attn_norm = (attn_arr - a_min) / (a_max - a_min + 1e-8)
    word_html = ""
    for w_txt, a in zip(words, attn_norm):
        alpha = 0.12 + 0.88 * float(a)
        word_html += (
            f'<span style="background:rgba(107,63,160,{alpha:.2f});'
            f'padding:1px 4px;border-radius:3px;margin:1px;font-size:12px">'
            f'{w_txt}</span> '
        )

    step1 = f"""
  <div style="font-size:10px;color:#bbb;text-transform:uppercase;
              letter-spacing:1.2px;margin-bottom:6px">
    🧠 Step 1 — BiLSTM flavor encoding
  </div>
  <div style="font-size:12px;color:#888;margin-bottom:6px">
    Flavor description for <em>{display}</em> tokenised and fed through the trained BiLSTM:
  </div>
  <div style="background:#f0ebe0;padding:10px 14px;border-radius:8px;
              line-height:1.9;margin-bottom:14px">
    {word_html}
    <div style="font-size:10px;color:#bbb;margin-top:6px">
      Word opacity = attention weight (darker = more attended)
    </div>
  </div>
"""

    # ── Stage 1: show attention map, spinner for clusters ────────────────────
    spinner2 = (
        '<div style="padding:16px;text-align:center;color:#6b3fa0;font-size:14px">'
        '⏳ Computing cosine similarity to 9 flavor clusters…</div>'
    )
    yield WRAP_OPEN + header + step1 + spinner2 + FOOTER + '</div>'

    # Cluster similarity bars (top-5)
    sorted_k = np.argsort(sims)[::-1][:5] if len(sims) > 0 else []
    sim_bars = ""
    for k in sorted_k:
        w   = int(float(sims[k]) * 230)
        col = "#6b3fa0" if int(k) == cluster_idx else "#d4cfc8"
        fw  = "700" if int(k) == cluster_idx else "400"
        sim_bars += (
            f'<div style="display:flex;align-items:center;margin:2px 0;font-size:11px">'
            f'<span style="width:190px;overflow:hidden;white-space:nowrap;'
            f'color:#555;font-weight:{fw}">'
            f'{CLUSTER_NAMES.get(int(k), str(k))}</span>'
            f'<div style="background:{col};width:{w}px;height:10px;'
            f'border-radius:2px;margin:0 6px"></div>'
            f'<span style="color:#999;font-weight:{fw}">{float(sims[k]):.3f}</span>'
            f'</div>'
        )

    step2 = f"""
  <div style="font-size:10px;color:#bbb;text-transform:uppercase;
              letter-spacing:1.2px;margin-bottom:6px">
    🎯 Step 2 — Cosine similarity to 9 flavor clusters
  </div>
  {sim_bars}
  <div style="margin-top:10px;margin-bottom:4px;font-size:13px;
              font-weight:700;color:#6b3fa0">
    → Primary flavor cluster: <strong>{cluster_name}</strong>
  </div>
  <hr style="margin:18px 0;border:none;border-top:1px solid #e4ddd2">
"""

    # ── Stage 2: show clusters, spinner for wines ────────────────────────────
    spinner3 = (
        '<div style="padding:16px;text-align:center;color:#6b3fa0;font-size:14px">'
        '⏳ Selecting wine pairings from cluster pools…</div>'
    )
    yield WRAP_OPEN + header + step1 + step2 + spinner3 + FOOTER + '</div>'

    # Derive food feel from Safe Bet cluster
    safe_cluster = recs[0].get("name", cluster_name) if recs else cluster_name
    feel = _food_feel(safe_cluster)

    # Build tier cards (notebook style)
    tier_cards_html = ""
    if recs:
        for rec in recs[:3]:
            tier_cards_html += _tier_card_html(rec, display, feel)
    else:
        tier_cards_html = ('<div style="padding:20px;color:#bbb;text-align:center">'
                           'No pairing data available for this food.</div>')

    step3 = f"""
  <div style="font-size:10px;color:#bbb;text-transform:uppercase;
              letter-spacing:1.2px;margin-bottom:12px">
    🍷 Step 3 — Wine Pairings
  </div>
  {tier_cards_html}
"""

    # ── Stage 3: complete card ───────────────────────────────────────────────
    yield WRAP_OPEN + header + step1 + step2 + step3 + FOOTER + '</div>'


# ── App state ─────────────────────────────────────────────────────────────────
_state: dict = {"food": "", "conf": 0.0, "top5": []}

# ── Event handlers ─────────────────────────────────────────────────────────────
def on_identify(pil_img):
    if pil_img is None:
        return (
            "*Upload a food photo and click **Identify Food** to begin.*",
            gr.update(visible=False),
            gr.update(visible=False),
            "",
        )
    food_name, conf, top5 = identify_food(pil_img)
    _state.update(food=food_name, conf=conf, top5=top5)
    runner_up = (
        f"{top5[1][0]}  ({top5[1][1]*100:.0f}%)" if len(top5) > 1 else "—"
    )
    msg = (
        f"## 🔍 I think this is…\n\n"
        f"# {food_name}\n\n"
        f"I'm **{conf*100:.0f}% confident** in that.\n\n"
        f"*(Runner-up: {runner_up})*\n\n"
        f"---\n**Is that right?**"
    )
    return msg, gr.update(visible=True), gr.update(visible=False), ""

def on_yes():
    food_name = _state.get("food", "")
    food_key  = food_name.lower().replace(" ", "_")
    cluster, cluster_name, sims, desc, attn_w = bilstm_encode(food_key)
    for html in _wine_card_parts(food_name, _state["conf"], _state["top5"],
                                 cluster, cluster_name, sims, desc, attn_w):
        time.sleep(0.6)
        yield gr.update(visible=True), html

def on_no():
    return (
        "*Upload a different photo and click **Identify Food** to try again.*",
        gr.update(visible=False),
        gr.update(visible=False),
        "",
    )

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="orange"),
    title="Wine & Dine 🍷",
) as demo:

    gr.Markdown(
        "# 🍽️ Wine & Dine\n"
        "**Upload a food photo — we identify it and find your perfect wine pairing.**"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            img_input    = gr.Image(type="pil", label="📷 Food photo", height=340)
            identify_btn = gr.Button("🔍 Identify Food", variant="primary", size="lg")

        with gr.Column(scale=1, min_width=340):
            prediction_md = gr.Markdown(
                "*Upload a photo and click **Identify Food** to begin.*"
            )
            with gr.Row(visible=False) as confirm_row:
                yes_btn = gr.Button("✅  Yes — show me wine pairings!", variant="primary")
                no_btn  = gr.Button("❌  No, try another photo",        variant="secondary")

    with gr.Column(visible=False) as card_group:
        gr.Markdown("---\n## 🍷 Your Wine Pairings")
        wine_card = gr.HTML()

    # wiring
    identify_btn.click(
        on_identify,
        inputs=[img_input],
        outputs=[prediction_md, confirm_row, card_group, wine_card],
    )
    yes_btn.click(on_yes,  inputs=None, outputs=[card_group, wine_card])
    no_btn.click( on_no,   inputs=None, outputs=[prediction_md, confirm_row, card_group, wine_card])

if __name__ == "__main__":
    demo.launch()
