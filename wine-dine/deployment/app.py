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
import io
import base64
import json
import time
import string
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
BILSTM_WEIGHTS = os.path.join(WEIGHTS_DIR, "tastebilstm_best.pt")
DATA_JSON      = os.path.join(DATA_DIR,    "food_flavor_description_v2.json")
VOCAB_JSON     = os.path.join(DATA_DIR,    "vocab.json")
CLUSTER_JSON   = os.path.join(DATA_DIR,    "cluster_names.json")
RESULTS_JSON   = os.path.join(DATA_DIR,    "results_all.json")
CENTROIDS_NPY  = os.path.join(DATA_DIR,    "centroids.npy")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── BiLSTM hyperparameters (must match training) ──────────────────────────────
HIDDEN_DIM    = 256
N_LAYERS      = 2
DROPOUT_RNN   = 0.4
EMBED_DIM     = 100
MAX_SEQ_LEN   = 64
N_TASTE_AXES  = 10  # TasteBiLSTM output — 10 sensory axes

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

# TasteBiLSTM uses the same vocab.json (4294 tokens, MIN_FREQ=3)
VOCAB_SIZE = len(VOCAB)  # matches embedding.weight shape in tastebilstm_best.pt

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
                            N_TASTE_AXES, N_LAYERS, DROPOUT_RNN)
    if os.path.exists(BILSTM_WEIGHTS):
        ckpt = torch.load(BILSTM_WEIGHTS, map_location=DEVICE)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        model.load_state_dict(ckpt, strict=False)
    else:
        print(f"WARNING: {BILSTM_WEIGHTS} not found — using random weights.")
    return model.to(DEVICE).eval()

bilstm = _load_bilstm()


_PUNCT_STRIP = str.maketrans("", "", string.punctuation)

def _tokenize(text):
    # strip punctuation then lowercase — matches §7/§14.5 tokeniser exactly
    words  = str(text).lower().translate(_PUNCT_STRIP).split()
    tokens = [VOCAB.get(w, 1) for w in words[:MAX_SEQ_LEN]]  # OOV→1, not 0
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
        cent_l2 = CENTROIDS / (np.linalg.norm(CENTROIDS, axis=1, keepdims=True) + 1e-8)
        sims    = cent_l2 @ vec_l2
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

# Colours match mockup design tokens
# ── Design tokens (wine-editorial palette) ──────────────────────────────────
# Primary:   --wine       #7a1f32  deep wine red
# Secondary: --olive      #697a54  safe/nature
#            --lavender   #6961a8  hidden gem / subtle
#            --terracotta #bd6846  bold / warm contrast
# Background: --cream     #f8f3eb  warm off-white

_TIER_COLOR = {
    "SAFE BET":   "#697a54",   # olive
    "HIDDEN GEM": "#6961a8",   # lavender
    "BOLD MOVE":  "#bd6846",   # terracotta
}

_TIER_STRIP_BG = {
    "SAFE BET":   "rgba(105,122,84,0.12)",
    "HIDDEN GEM": "rgba(105,97,168,0.12)",
    "BOLD MOVE":  "rgba(189,104,70,0.12)",
}

_TIER_ICON = {
    "SAFE BET":   "✓",
    "HIDDEN GEM": "◆",
    "BOLD MOVE":  "↯",
}

_TIER_CONF_LABEL = {
    "SAFE BET":   "match",
    "HIDDEN GEM": "match",
    "BOLD MOVE":  "contrast",
}

_TIER_TAG_BG = {
    "SAFE BET":   "rgba(105,122,84,0.10)",
    "HIDDEN GEM": "rgba(105,97,168,0.10)",
    "BOLD MOVE":  "rgba(189,104,70,0.10)",
}

# Primary action colour used in shell nav, CTA buttons, active dots
_PRIMARY = "#7a1f32"


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


def _conf_bar_html(conf: float, color: str, label: str = "match") -> str:
    pct = int(conf * 100)
    bar_w = int(conf * 110)
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        f'<div style="background:rgba(64,42,31,0.10);border-radius:999px;width:110px;height:6px;overflow:hidden">'
        f'<div style="background:{color};width:{bar_w}px;height:6px;border-radius:999px"></div>'
        f'</div>'
        f'<span style="font-size:22px;font-weight:900;color:{color};line-height:1;font-family:Georgia,serif">{pct}%</span>'
        f'<span style="font-size:10px;color:#b8aaa0;font-weight:800;text-transform:uppercase;letter-spacing:0.08em">{label}</span>'
        f'</div>'
    )


# ── HTML builders ─────────────────────────────────────────────────────────────
def _top5_bars_html(top5, confirmed_food):
    html = ""
    for fn, fp in top5:
        w   = int(fp * 250)
        col = "#7a1f32" if fn == confirmed_food else "#ddd5cc"
        fw  = "900"    if fn == confirmed_food else "400"
        html += (
            f'<div style="display:flex;align-items:center;margin:4px 0;font-size:12px">'
            f'<span style="width:200px;overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap;color:#4a1020;font-weight:{fw}">{fn}</span>'
            f'<div style="background:{col};width:{w}px;height:7px;'
            f'border-radius:999px;margin:0 8px"></div>'
            f'<span style="color:#756b63;font-weight:{fw}">{fp*100:.0f}%</span>'
            f'</div>'
        )
    return html


def _tier_card_html(rec: dict, display_name: str, feel: str, conf: float = 0.0) -> str:
    """Build one tier card — matches mockup design."""
    tier    = rec.get("tier", "")
    name    = rec.get("name", "")
    wine    = rec.get("wine", "—")
    rating  = rec.get("rating", "—")
    snippet = _clip(rec.get("snippet", ""), 120)
    kws     = rec.get("keywords", [])

    color    = _TIER_COLOR.get(tier, "#555")
    strip_bg = _TIER_STRIP_BG.get(tier, "#f5f5f5")
    icon     = _TIER_ICON.get(tier, "")
    intent   = _INTENT.get(tier, "pairs with")
    conf_lbl = _TIER_CONF_LABEL.get(tier, "match")
    tag_bg   = _TIER_TAG_BG.get(tier, "#eee")
    adj      = _cluster_adj(name)
    tier_lbl = tier.lower()

    tags_html = ""
    for i, kw in enumerate(kws[:3]):
        if i > 0:
            tags_html += '<span style="font-size:10px;color:#aaa">·</span>'
        tags_html += (
            f'<span style="font-size:10px;font-weight:500;padding:2px 7px;'
            f'border-radius:20px;background:{tag_bg};color:{color}">{kw}</span>'
        )

    reasoning = (
        f'Your {display_name.lower()} is {feel} — '
        + ("this wine matches that energy exactly." if tier == "SAFE BET"
           else "this wine finds an angle most pairings overlook." if tier == "HIDDEN GEM"
           else "this wine goes against it entirely. Sometimes contrast is the pairing.")
    )

    return f"""
    <div style="border-radius:12px;overflow:hidden;background:#fff;
                border:0.5px solid rgba(0,0,0,0.08);margin:0 0 10px 0;
                box-shadow:0 1px 4px rgba(0,0,0,0.05)">
      <div style="background:{strip_bg};padding:9px 12px;
                  display:flex;justify-content:space-between;align-items:flex-start">
        <div style="font-size:10px;font-weight:600;letter-spacing:0.05em;
                    color:{color};display:flex;align-items:center;gap:4px">
          {icon} {tier_lbl}
        </div>
        <div>
          {_conf_bar_html(conf, color, conf_lbl)}
        </div>
      </div>
      <div style="padding:10px 12px;display:flex;flex-direction:column;gap:8px">
        <div style="font-size:11px;color:#666;font-style:italic;line-height:1.55">
          {reasoning}
        </div>
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
          <span style="font-size:10px;color:#aaa">plays on</span>
          {tags_html}
        </div>
        <div style="height:0.5px;background:rgba(0,0,0,0.07)"></div>
        <div style="font-size:12px;font-weight:600;color:#1a1917;line-height:1.35">{wine}</div>
        <div style="display:flex;align-items:center;gap:4px">
          <span style="color:#E6A817;font-size:11px">★</span>
          <span style="font-size:11px;font-weight:500;color:#1a1917">{rating}</span>
          <span style="font-size:10px;color:#aaa">/ 100</span>
        </div>
        <div style="font-size:10.5px;color:#666;font-style:italic;line-height:1.5;
                    border-left:2px solid rgba(0,0,0,0.08);padding-left:7px">
          "{snippet}"
        </div>
      </div>
    </div>"""


def _screen1_html(food_name: str, conf: float, top5: list, img_b64: str = "") -> str:
    """Screen 1 (step 2) — dish detection result matching mockup 'Is this [dish]?' layout."""
    display  = food_name.replace("_", " ").title()
    conf_pct = int(conf * 100)
    conf_bar = int(conf * 100)   # % used directly in CSS width

    # Photo — real image or placeholder
    if img_b64:
        photo_el = (
            f'<img src="data:image/jpeg;base64,{img_b64}"'
            f' style="width:100%;height:520px;object-fit:cover;display:block;'
            f'border-radius:20px">'
        )
    else:
        photo_el = (
            '<div style="width:100%;height:520px;display:flex;align-items:center;'
            'justify-content:center;font-size:72px;border-radius:20px;'
            'background:linear-gradient(135deg,#ede3d8,#ddd0c4)">🍽️</div>'
        )

    # Other possibilities (top-5 positions 2–5, skipping top-1)
    others_rows = ""
    for fn, fp in top5[1:]:
        bar_pct = int(fp * 100)
        others_rows += (
            f'<div style="display:grid;grid-template-columns:1fr auto;'
            f'gap:10px;align-items:center;margin-bottom:10px">'
            f'<div>'
            f'<div style="font-size:12px;color:#756b63;margin-bottom:4px;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{fn}</div>'
            f'<div style="height:5px;background:rgba(64,42,31,0.10);'
            f'border-radius:999px;overflow:hidden">'
            f'<div style="background:#c9a15d;width:{bar_pct}%;height:100%;'
            f'border-radius:999px"></div>'
            f'</div>'
            f'</div>'
            f'<span style="font-size:12px;color:#9e9188;font-weight:700;'
            f'white-space:nowrap">{fp*100:.1f}%</span>'
            f'</div>'
        )

    return f"""
<div style="font-family:'Segoe UI',system-ui,Arial,sans-serif">

  <!-- step label -->
  <div style="font-size:11px;color:#7a1830;text-transform:uppercase;
              letter-spacing:0.14em;font-weight:800;margin-bottom:10px">
    STEP 2 &middot; DETECT DISH</div>

  <!-- heading -->
  <div style="font-family:Georgia,'Times New Roman',serif;font-size:38px;
              font-weight:700;color:#211917;letter-spacing:-1.5px;
              line-height:1;margin-bottom:6px">Is this {display}?</div>
  <div style="font-size:13px;color:#9e9188;margin-bottom:24px">
    Please verify the detected dish</div>

  <!-- 2-col: photo | info panel -->
  <div style="display:grid;grid-template-columns:1.1fr 0.9fr;gap:28px;align-items:start">

    <!-- photo -->
    <div style="border-radius:20px;overflow:hidden;
                box-shadow:0 18px 48px rgba(52,34,26,0.14)">
      {photo_el}
    </div>

    <!-- right panel -->
    <div style="display:flex;flex-direction:column;gap:0">

      <!-- "we think this is" label -->
      <div style="font-size:10px;color:#b8aaa0;text-transform:uppercase;
                  letter-spacing:0.14em;font-weight:800;margin-bottom:8px">
        we think this is</div>

      <!-- dish name -->
      <div style="font-family:Georgia,'Times New Roman',serif;
                  font-size:52px;line-height:0.88;letter-spacing:-2px;
                  color:#42101d;margin-bottom:20px">{display}</div>

      <!-- confidence -->
      <div style="margin-bottom:22px">
        <div style="display:flex;justify-content:space-between;
                    align-items:baseline;margin-bottom:7px">
          <span style="font-size:11px;color:#9e9188;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.1em">Confidence</span>
          <span style="font-family:Georgia,serif;font-size:28px;font-weight:700;
                       color:#7a1830;letter-spacing:-1px;line-height:1">{conf_pct}%</span>
        </div>
        <div style="height:8px;background:rgba(64,42,31,0.10);
                    border-radius:999px;overflow:hidden">
          <div style="background:linear-gradient(90deg,#8d1f3a,#c9536e);
                      width:{conf_bar}%;height:100%;border-radius:999px"></div>
        </div>
      </div>

      <!-- divider -->
      <div style="height:1px;background:rgba(63,43,35,0.09);margin-bottom:18px"></div>

      <!-- other possibilities -->
      <div style="font-size:10px;color:#b8aaa0;text-transform:uppercase;
                  letter-spacing:0.12em;font-weight:800;margin-bottom:12px">
        other possibilities</div>
      {others_rows}

    </div>
  </div>

</div>"""


_DETECT_HEADER_HTML = """
<div style="display:grid;grid-template-columns:auto 1fr auto;
            align-items:start;gap:24px;padding:24px 34px 18px;
            border-bottom:1px solid rgba(63,43,35,0.09)">
  <div>
    <div style="font-family:Georgia,'Times New Roman',serif;font-size:30px;
                letter-spacing:-0.8px;font-weight:700;white-space:nowrap;color:#211917">
      Wine<span style="color:#7a1830">&amp;</span>Dine</div>
    <div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
                color:#42101d;font-weight:800;margin-top:6px">Food &amp; wine pairing</div>
  </div>
  <div style="display:flex;align-items:flex-start;justify-content:center;gap:0;width:100%">
    <!-- Step 1 done -->
    <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
      <div style="width:48px;height:48px;border-radius:999px;display:grid;
                  place-items:center;font-size:16px;font-weight:900;
                  background:rgba(255,255,255,0.82);color:#756b63;
                  border:1px solid rgba(64,42,31,0.16)">&#10003;</div>
      <div style="font-size:12px;font-weight:800;color:#756b63;white-space:nowrap">Upload</div>
    </div>
    <div style="flex:1;height:1px;background:rgba(64,42,31,0.18);min-width:24px;margin-bottom:22px;align-self:center"></div>
    <!-- Step 2 active -->
    <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
      <div style="width:48px;height:48px;border-radius:999px;display:grid;
                  place-items:center;background:linear-gradient(135deg,#8d1f3a,#5a1024);
                  color:#fff;font-size:17px;font-weight:900;
                  box-shadow:0 10px 28px rgba(122,24,48,0.24)">2</div>
      <div style="font-size:12px;font-weight:800;color:#42101d;
                  text-decoration:underline;text-underline-offset:4px;white-space:nowrap">Detect Dish</div>
    </div>
    <div style="flex:1;height:1px;background:rgba(64,42,31,0.18);min-width:24px;margin-bottom:22px;align-self:center"></div>
    <!-- Step 3 -->
    <div style="display:flex;flex-direction:column;align-items:center;gap:8px;opacity:0.5">
      <div style="width:48px;height:48px;border-radius:999px;display:grid;
                  place-items:center;background:rgba(255,255,255,0.58);
                  color:#9e9188;font-size:17px;font-weight:900;
                  border:1px solid rgba(200,190,180,0.5)">3</div>
      <div style="font-size:12px;font-weight:800;color:#9e9188;white-space:nowrap">Fingerprint</div>
    </div>
    <div style="flex:1;height:1px;background:rgba(64,42,31,0.18);min-width:24px;margin-bottom:22px;align-self:center"></div>
    <!-- Step 4 -->
    <div style="display:flex;flex-direction:column;align-items:center;gap:8px;opacity:0.5">
      <div style="width:48px;height:48px;border-radius:999px;display:grid;
                  place-items:center;background:rgba(255,255,255,0.58);
                  color:#9e9188;font-size:17px;font-weight:900;
                  border:1px solid rgba(200,190,180,0.5)">4</div>
      <div style="font-size:12px;font-weight:800;color:#9e9188;white-space:nowrap">Wine Pairings</div>
    </div>
    <div style="flex:1;height:1px;background:rgba(64,42,31,0.18);min-width:24px;margin-bottom:22px;align-self:center"></div>
    <!-- Step 5 -->
    <div style="display:flex;flex-direction:column;align-items:center;gap:8px;opacity:0.5">
      <div style="width:48px;height:48px;border-radius:999px;display:grid;
                  place-items:center;background:rgba(255,255,255,0.58);
                  color:#9e9188;font-size:17px;font-weight:900;
                  border:1px solid rgba(200,190,180,0.5)">5</div>
      <div style="font-size:12px;font-weight:800;color:#9e9188;white-space:nowrap">The Story</div>
    </div>
  </div>
  <div style="width:140px;display:flex;align-items:center;justify-content:center">
    <div style="background:linear-gradient(135deg,#c0334d 0%,#8d1f3a 100%);color:#fff;
               border-radius:999px;font-size:13px;font-weight:800;
               padding:10px 20px;white-space:nowrap;user-select:none;
               box-shadow:0 8px 22px rgba(122,24,48,0.24);text-align:center">
      &#x21BA;&nbsp;&nbsp;Start Over
    </div>
  </div>
</div>
"""


def _s1_head_html(food_name: str) -> str:
    display = food_name.replace("_", " ").title() if "_" in food_name else food_name
    return (
        f'<div style="font-family:\'Segoe UI\',system-ui,Arial,sans-serif;margin-bottom:22px">'
        f'<div style="font-size:12px;color:#7a1830;text-transform:uppercase;'
        f'letter-spacing:0.14em;font-weight:800;margin-bottom:10px">'
        f'STEP 2 &middot; DETECT DISH</div>'
        f'<div style="font-family:Georgia,\'Times New Roman\',serif;font-size:40px;'
        f'font-weight:700;color:#211917;letter-spacing:-1.4px;line-height:1;'
        f'margin-bottom:8px">Is this {display}?</div>'
        f'<div style="font-size:13px;color:#9e9188;line-height:1.6">Please verify the detected dish</div>'
        f'</div>'
    )


def _s1_photo_html(img_b64: str) -> str:
    if img_b64:
        inner = (
            f'<img src="data:image/jpeg;base64,{img_b64}"'
            f' style="width:100%;height:520px;object-fit:cover;display:block;border-radius:20px">'
        )
    else:
        inner = (
            '<div style="width:100%;height:520px;display:flex;align-items:center;'
            'justify-content:center;font-size:72px;border-radius:20px;'
            'background:linear-gradient(135deg,#ede3d8,#ddd0c4)">🍽️</div>'
        )
    return (
        f'<div style="border-radius:20px;overflow:hidden;'
        f'box-shadow:0 18px 48px rgba(52,34,26,0.14)">{inner}</div>'
    )


def _s1_info_html(food_name: str, conf: float, top5: list) -> str:
    display  = food_name.replace("_", " ").title() if "_" in food_name else food_name
    conf_pct = int(conf * 100)
    others   = ""
    for fn, fp in top5[1:]:
        bar_pct = int(fp * 100)
        others += (
            f'<div style="display:grid;grid-template-columns:1fr auto;'
            f'gap:10px;align-items:center;margin-bottom:10px">'
            f'<div>'
            f'<div style="font-size:12px;color:#756b63;margin-bottom:4px;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{fn}</div>'
            f'<div style="height:5px;background:rgba(64,42,31,0.10);'
            f'border-radius:999px;overflow:hidden">'
            f'<div style="background:#c9a15d;width:{bar_pct}%;height:100%;'
            f'border-radius:999px"></div></div></div>'
            f'<span style="font-size:12px;color:#9e9188;font-weight:700;'
            f'white-space:nowrap">{fp*100:.1f}%</span></div>'
        )
    return (
        f'<div style="font-family:\'Segoe UI\',system-ui,Arial,sans-serif;'
        f'display:flex;flex-direction:column;gap:0">'
        f'<div style="font-size:11px;color:#b8aaa0;text-transform:uppercase;'
        f'letter-spacing:0.14em;font-weight:800;margin-bottom:8px">we think this is</div>'
        f'<div style="font-family:Georgia,\'Times New Roman\',serif;font-size:40px;'
        f'font-weight:700;line-height:1;letter-spacing:-1.4px;color:#42101d;margin-bottom:22px">{display}</div>'
        f'<div style="margin-bottom:22px">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:7px">'
        f'<span style="font-size:11px;color:#9e9188;font-weight:800;text-transform:uppercase;'
        f'letter-spacing:0.12em">Confidence</span>'
        f'<span style="font-family:Georgia,serif;font-size:26px;font-weight:700;'
        f'color:#7a1830;letter-spacing:-0.8px;line-height:1">{conf_pct}%</span></div>'
        f'<div style="height:8px;background:rgba(64,42,31,0.10);border-radius:999px;overflow:hidden">'
        f'<div style="background:linear-gradient(90deg,#8d1f3a,#c9536e);'
        f'width:{conf_pct}%;height:100%;border-radius:999px"></div></div></div>'
        f'<div style="height:1px;background:rgba(63,43,35,0.09);margin-bottom:18px"></div>'
        f'<div style="font-size:11px;color:#b8aaa0;text-transform:uppercase;'
        f'letter-spacing:0.12em;font-weight:800;margin-bottom:12px">other possibilities</div>'
        f'{others}</div>'
    )


def _screen2_html(display: str, desc: str, attn_w,
                  cluster_idx: int, cluster_name: str, sims,
                  img_b64: str = "") -> str:
    """Screen 2 (step 3) — taste fingerprint: photo left, heatmap + bars right."""

    # ── Attention-highlighted word chips (continuous gradient — every word colored) ──
    words    = desc.split()[:MAX_SEQ_LEN]
    attn_arr = attn_w[:len(words)]
    a_min, a_max = attn_arr.min(), attn_arr.max()
    attn_norm = (attn_arr - a_min) / (a_max - a_min + 1e-8)
    # interpolate background from light cream (low attention) → wine red (high attention)
    # `a_visual = a ** 0.55` boosts low-attention values so every word stays visible
    word_html = ""
    for w_txt, a in zip(words, attn_norm):
        a_visual = float(a) ** 0.55
        r = int(248 + (122 - 248) * a_visual)
        g = int(238 + (24  - 238) * a_visual)
        b = int(228 + (48  - 228) * a_visual)
        text_color = "#fff" if a_visual > 0.48 else "#42101d"
        weight = 600 if a_visual < 0.30 else (800 if a_visual > 0.65 else 700)
        chip = f"background:rgb({r},{g},{b});color:{text_color};font-weight:{weight}"
        word_html += (
            f'<span style="display:inline-block;border-radius:6px;'
            f'padding:3px 8px;margin:2px 2px;font-size:13px;'
            f'line-height:1.45;cursor:default;{chip}">{w_txt}</span>'
        )

    # ── Cluster similarity bars (top 3 only) ─────────────────────────────────
    sorted_k = np.argsort(sims)[::-1][:3] if len(sims) > 0 else []
    cluster_rows = ""
    for k in sorted_k:
        sim_val = float(sims[k])
        bar_pct = int(sim_val * 100)
        is_top  = int(k) == cluster_idx
        name_c  = "#42101d" if is_top else "#9e9188"
        name_fw = "700"     if is_top else "500"
        bar_c   = "#8d1f3a" if is_top else "#ddd5c8"
        lbl     = CLUSTER_NAMES.get(int(k), str(k))
        cluster_rows += (
            f'<div style="display:grid;grid-template-columns:1fr auto;'
            f'gap:12px;align-items:center;margin-bottom:12px">'
            f'<div>'
            f'<div style="font-size:12px;color:{name_c};font-weight:{name_fw};'
            f'margin-bottom:5px;overflow:hidden;white-space:nowrap;'
            f'text-overflow:ellipsis">{lbl}</div>'
            f'<div style="height:5px;background:rgba(64,42,31,0.09);'
            f'border-radius:999px;overflow:hidden">'
            f'<div style="background:{bar_c};width:{bar_pct}%;height:100%;'
            f'border-radius:999px"></div>'
            f'</div>'
            f'</div>'
            f'<span style="font-size:12px;color:{name_c};font-weight:{name_fw};'
            f'white-space:nowrap">{sim_val:.2f}</span>'
            f'</div>'
        )

    # ── Food photo (left column) ──────────────────────────────────────────────
    if img_b64:
        photo_el = (
            f'<img src="data:image/jpeg;base64,{img_b64}"'
            f' style="width:100%;height:100%;object-fit:cover;display:block;'
            f'border-radius:16px">'
        )
    else:
        photo_el = (
            '<div style="width:100%;height:100%;min-height:260px;display:flex;'
            'align-items:center;justify-content:center;font-size:56px;'
            'border-radius:16px;background:linear-gradient(135deg,#ede3d8,#ddd0c4)">🍽️</div>'
        )

    return f"""
<div style="font-family:'Segoe UI',system-ui,Arial,sans-serif">

  <!-- step label + heading (same pattern as every other screen) -->
  <div style="font-size:12px;color:#7a1830;text-transform:uppercase;
              letter-spacing:0.14em;font-weight:800;margin-bottom:10px">
    STEP 3 &middot; TASTE FINGERPRINT</div>
  <div style="font-family:Georgia,'Times New Roman',serif;font-size:40px;
              font-weight:700;color:#211917;letter-spacing:-1.4px;
              line-height:1;margin-bottom:8px">How does this {display} taste?</div>
  <div style="font-size:13px;color:#9e9188;line-height:1.6;margin-bottom:24px">
    We turn the verified dish into a flavor profile the wine matcher can use.</div>

  <!-- 2-col layout: photo LEFT | badge + bars RIGHT -->
  <div style="display:grid;grid-template-columns:1fr 0.85fr;gap:28px;
              align-items:start;margin-bottom:22px">

    <!-- LEFT: food photo -->
    <div style="border-radius:16px;overflow:hidden;
                box-shadow:0 12px 32px rgba(52,34,26,0.12);
                aspect-ratio:4/3">
      {photo_el}
    </div>

    <!-- RIGHT: badge + cluster bars -->
    <div style="display:flex;flex-direction:column;gap:18px">

      <!-- taste fingerprint badge -->
      <div style="padding:14px 16px;background:rgba(122,24,48,0.06);
                  border-radius:14px;border-left:3px solid #7a1830">
        <div style="font-size:11px;color:#b8aaa0;text-transform:uppercase;
                    letter-spacing:0.12em;font-weight:800;margin-bottom:4px">
          Flavor profile</div>
        <div style="font-family:Georgia,'Times New Roman',serif;font-size:22px;
                    font-weight:700;color:#42101d;letter-spacing:-0.5px;
                    line-height:1.2">{cluster_name}</div>
      </div>

      <!-- cluster similarity bars -->
      <div>
        <div style="font-size:11px;color:#b8aaa0;text-transform:uppercase;
                    letter-spacing:0.12em;font-weight:800;margin-bottom:12px">
          Closest flavor worlds</div>
        {cluster_rows}
      </div>

    </div>
  </div>

  <!-- full-width attention heatmap with gradient coloring -->
  <div style="margin-bottom:22px">
    <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:8px">
      <div style="font-size:11px;color:#9e9188;text-transform:uppercase;
                  letter-spacing:0.12em;font-weight:800">Attention heatmap</div>
      <div style="font-size:10px;color:#b8aaa0;display:flex;align-items:center;gap:8px">
        <span>less</span>
        <span style="display:inline-block;width:80px;height:8px;border-radius:4px;
                     background:linear-gradient(90deg,rgb(248,238,228),rgb(122,24,48))"></span>
        <span>more</span>
      </div>
    </div>
    <div style="background:rgba(251,247,241,0.8);border-radius:14px;
                padding:14px 16px;border:1px solid rgba(63,43,35,0.09);
                line-height:1.6">
      {word_html}
    </div>
    <div style="font-size:10px;color:#b8aaa0;margin-top:6px">
      Word warmth = Bahdanau attention weight &middot; darker = more influential</div>
  </div>

  <!-- CTA -->
  <div style="text-align:center">
    <label for="wdt2"
           style="display:inline-block;padding:13px 36px;border-radius:999px;
                  background:linear-gradient(135deg,#8d1f3a,#5a1024);
                  color:#fff;font-size:14px;font-weight:900;
                  cursor:pointer;letter-spacing:-0.2px;
                  box-shadow:0 12px 28px rgba(122,24,48,0.24)">
      Find wine pairings &rarr;
    </label>
  </div>

</div>"""


def _screen4_html() -> str:
    """Screen 4 (step 5) -- The Story: project narrative + presentation placeholder."""
    return """
<div style="font-family:'Segoe UI',system-ui,Arial,sans-serif">

  <!-- step label + heading (same pattern as every other screen) -->
  <div style="margin-bottom:24px">
    <div style="font-size:12px;color:#7a1830;text-transform:uppercase;
                letter-spacing:0.14em;font-weight:800;margin-bottom:10px">
      STEP 5 &middot; THE STORY</div>
    <div style="font-family:Georgia,'Times New Roman',serif;font-size:40px;
                font-weight:700;color:#211917;letter-spacing:-1.4px;
                line-height:1;margin-bottom:8px">The Story of Wine&amp;Dine</div>
    <div style="font-size:13px;color:#9e9188;line-height:1.6;max-width:680px">
      Wine&amp;Dine started as an attempt to make wine pairing feel less intimidating
      and more intuitive. Instead of asking users to understand wine terminology,
      the system starts from something people already understand naturally: food.
    </div>
  </div>

  <!-- problem / approach cards -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:28px">
    <div style="padding:20px 22px;border-radius:18px;background:#fff;
                border:1px solid rgba(63,43,35,0.09);
                box-shadow:0 2px 8px rgba(52,34,26,0.06)">
      <div style="font-size:10px;color:#9e9188;text-transform:uppercase;
                  letter-spacing:0.14em;font-weight:800;margin-bottom:8px">PROBLEM</div>
      <div style="font-family:Georgia,'Times New Roman',serif;font-size:18px;
                  font-weight:700;color:#211917;line-height:1.3;margin-bottom:10px">
        Wine recommendations are overwhelming</div>
      <div style="font-size:13px;color:#7c726b;line-height:1.7">
        Most pairing experiences expect users to understand grape regions,
        tannins and wine language before they can make a confident choice.
      </div>
    </div>
    <div style="padding:20px 22px;border-radius:18px;background:#fff;
                border:1px solid rgba(63,43,35,0.09);
                box-shadow:0 2px 8px rgba(52,34,26,0.06)">
      <div style="font-size:10px;color:#9e9188;text-transform:uppercase;
                  letter-spacing:0.14em;font-weight:800;margin-bottom:8px">APPROACH</div>
      <div style="font-family:Georgia,'Times New Roman',serif;font-size:18px;
                  font-weight:700;color:#211917;line-height:1.3;margin-bottom:10px">
        Start from taste, not wine knowledge</div>
      <div style="font-size:13px;color:#7c726b;line-height:1.7">
        The app translates food into flavor understanding first, then builds
        recommendations around similarity, discovery and contrast.
      </div>
    </div>
  </div>

  <!-- presentation placeholder -->
  <div style="border-radius:18px;border:1.5px dashed rgba(122,24,48,0.22);
              background:rgba(122,24,48,0.03);padding:32px;text-align:center;
              margin-bottom:28px">
    <div style="font-size:10px;color:#b8aaa0;text-transform:uppercase;
                letter-spacing:0.16em;font-weight:800;margin-bottom:10px">
      PRESENTATION PLACEHOLDER</div>
    <div style="font-family:Georgia,'Times New Roman',serif;font-size:28px;
                font-weight:700;color:#7a1830;letter-spacing:-0.8px;
                margin-bottom:10px">Project Presentation</div>
    <div style="font-size:13px;color:#9e9188;margin-bottom:20px">
      Add link to PowerPoint / research presentation / architecture deck here.</div>
    <a href="#" style="display:inline-block;padding:12px 28px;border-radius:999px;
                       background:linear-gradient(135deg,#8d1f3a,#5a1024);
                       color:#fff;font-size:14px;font-weight:800;
                       text-decoration:none;letter-spacing:-0.2px;
                       box-shadow:0 10px 24px rgba(122,24,48,0.22)">
      Open presentation &#x2197;</a>
  </div>

  <!-- tags -->
  <div style="text-align:center;font-size:12px;color:#b8aaa0">
    Machine Learning &middot; Recommendation Systems &middot; Explainable AI &middot; UX Design
  </div>

</div>"""

_TIER_INFO = {
    "SAFE BET":   "nearest cluster centroid by cosine similarity to dominant flavor description",
    "HIDDEN GEM": "nearest centroid to the \u2018surprising pairing\u2019 flavor description",
    "BOLD MOVE":  "score = distance \u00d7 (1 + secondary affinity) \u00b7 maximises contrast while keeping the wine drinkable",
}
_WINE_INFO = "highest cosine similarity between food taste vector and wine taste vectors within this cluster\u2019s 10-wine pool"


def _screen3_html(display: str, cluster_name: str, recs: list, feel: str) -> str:
    """Screen 3 (step 4) -- wine pairings. Zero JS."""
    import re as _re

    def _extract_year(wine_str):
        m = _re.search(r'\b(19|20)\d{2}\b', str(wine_str))
        return m.group(0) if m else ""

    def _clean_wine_name(wine_str):
        return _re.sub(r'\s*(19|20)\d{2}\.?\d*\s*$', '', str(wine_str)).strip()

    cards_html = ""
    for rec in recs[:3]:
        tier     = rec.get("tier", "")
        wine     = rec.get("wine", "—")
        snippet  = _clip(rec.get("snippet", ""), 140)
        kws      = rec.get("keywords", [])
        conf     = float(rec.get("confidence", 0.0))

        color    = _TIER_COLOR.get(tier, "#555")
        strip_bg = _TIER_STRIP_BG.get(tier, "#f5f5f5")
        icon     = _TIER_ICON.get(tier, "")
        conf_lbl = _TIER_CONF_LABEL.get(tier, "match")
        tag_bg   = _TIER_TAG_BG.get(tier, "#eee")
        tier_lbl = tier.lower()
        conf_pct = int(conf * 100)

        wine_name = _clean_wine_name(wine)
        wine_year = _extract_year(wine)
        year_html = (f'<div style="font-size:12px;color:#9e9188;margin-top:2px">'
                     f'{wine_year}</div>') if wine_year else ""

        reasoning = (
            "This wine matches your dish's energy exactly." if tier == "SAFE BET"
            else "This wine finds an angle most pairings overlook." if tier == "HIDDEN GEM"
            else "A deliberate contrast pairing — goes against the dish. That's the point."
        )

        tags_html = ""
        for kw in kws[:5]:
            tags_html += (
                f'<span style="font-size:11px;font-weight:600;padding:4px 10px;'
                f'border-radius:20px;background:{tag_bg};color:{color};'
                f'white-space:nowrap">{kw}</span>'
            )

        cards_html += f"""
<div style="border-radius:18px;overflow:hidden;background:#fff;
            border:1px solid rgba(63,43,35,0.09);
            box-shadow:0 4px 16px rgba(52,34,26,0.07)">

  <!-- tier header: badge left, % right -->
  <div style="background:{strip_bg};padding:14px 16px;
              display:flex;align-items:center;justify-content:space-between">
    <div style="display:flex;align-items:center;gap:6px">
      <span style="font-size:12px;font-weight:900;color:{color}">{icon}</span>
      <span style="font-size:11px;font-weight:800;letter-spacing:0.08em;
                   text-transform:uppercase;color:{color}">{tier_lbl}</span>
    </div>
    <div style="text-align:right">
      <div style="font-family:Georgia,serif;font-size:26px;font-weight:700;
                  color:#7a1830;line-height:1;letter-spacing:-0.5px">{conf_pct}%</div>
      <div style="font-size:9px;color:#b8aaa0;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.08em">{conf_lbl}</div>
    </div>
  </div>

  <!-- card body -->
  <div style="padding:16px;display:flex;flex-direction:column;gap:10px">

    <!-- wine name + year -->
    <div>
      <div style="font-family:Georgia,'Times New Roman',serif;font-size:20px;
                  font-weight:700;color:#211917;line-height:1.2">{wine_name}</div>
      {year_html}
    </div>

    <!-- pairing reasoning -->
    <div style="font-size:13px;color:#7c726b;line-height:1.6">{reasoning}</div>

    <!-- flavor notes -->
    <div style="display:flex;flex-wrap:wrap;gap:5px">{tags_html}</div>

    <!-- divider -->
    <div style="height:1px;background:rgba(63,43,35,0.09)"></div>

    <!-- review quote -->
    <div style="font-size:12px;color:#7c726b;font-style:italic;line-height:1.7;
                border-left:2px solid {color};padding-left:10px">
      &ldquo;{snippet}&rdquo;
    </div>

  </div>
</div>"""

    return f"""
<div style="font-family:'Segoe UI',system-ui,Arial,sans-serif">

  <!-- step label + heading (same pattern as every other screen) -->
  <div style="font-size:12px;color:#7a1830;text-transform:uppercase;
              letter-spacing:0.14em;font-weight:800;margin-bottom:10px">
    STEP 4 &middot; WINE PAIRINGS</div>
  <div style="font-family:Georgia,'Times New Roman',serif;font-size:40px;
              font-weight:700;color:#211917;letter-spacing:-1.4px;
              line-height:1;margin-bottom:24px">Three wines, three directions</div>

  <!-- 3-col card grid -->
  <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
              gap:16px;margin-bottom:8px">
    {cards_html}
  </div>

</div>"""

def _shell_html(s1: str, s2: str, s3: str, cur: int, s4: str = "") -> str:
    """
    Full 4-screen shell: 5-step header, glassmorphism card, CSS radio-tab nav — zero JS.
    Screens: 0=Detect Dish (step 2), 1=Taste Fingerprint (step 3),
             2=Wine Pairings (step 4), 3=The Story (step 5).
    cur = 0/1/2/3 — which tab is active on server render.

    IMPORTANT: #wds0/1/2/3 are DIRECT children of #wdshell (siblings of radio inputs)
    so that `#wdt0:checked ~ #wds0 { display:block }` works correctly.
    """
    unlocked = int(bool(s1)) + int(bool(s2)) + int(bool(s3)) - 1
    c = [" checked" if i == cur else "" for i in range(4)]
    _lock = "pointer-events:none;opacity:0.32"

    # ── CSS ───────────────────────────────────────────────────────────────────
    css = (
        "<style>"
        # Screen visibility — wds0/1/2 are DIRECT siblings of radio inputs ✓
        "#wds0,#wds1,#wds2,#wds3{display:none}"
        "#wdt0:checked~#wds0,"
        "#wdt1:checked~#wds1,"
        "#wdt2:checked~#wds2,"
        "#wdt3:checked~#wds3{display:block}"
        # Active step circle (wine-red gradient)
        "#wdt0:checked~#wdhdr .wds2circ,"
        "#wdt1:checked~#wdhdr .wds3circ,"
        "#wdt2:checked~#wdhdr .wds4circ,"
        "#wdt3:checked~#wdhdr .wds5circ"
        "{background:linear-gradient(135deg,#8d1f3a,#5a1024)!important;"
        "color:#fff!important;border-color:transparent!important;"
        "box-shadow:0 10px 28px rgba(122,24,48,0.24)!important}"
        # Done step circle (muted white)
        "#wdt1:checked~#wdhdr .wds2circ,"
        "#wdt2:checked~#wdhdr .wds2circ,"
        "#wdt2:checked~#wdhdr .wds3circ,"
        "#wdt3:checked~#wdhdr .wds2circ,"
        "#wdt3:checked~#wdhdr .wds3circ,"
        "#wdt3:checked~#wdhdr .wds4circ"
        "{background:rgba(255,255,255,0.82)!important;"
        "color:#756b63!important;border:1px solid rgba(64,42,31,0.16)!important;"
        "box-shadow:none!important}"
        # Active step label (underlined)
        "#wdt0:checked~#wdhdr .wds2lbl,"
        "#wdt1:checked~#wdhdr .wds3lbl,"
        "#wdt2:checked~#wdhdr .wds4lbl,"
        "#wdt3:checked~#wdhdr .wds5lbl"
        "{color:#42101d!important;text-decoration:underline;"
        "text-underline-offset:4px}"
        # Done step label (muted)
        "#wdt1:checked~#wdhdr .wds2lbl,"
        "#wdt2:checked~#wdhdr .wds2lbl,"
        "#wdt2:checked~#wdhdr .wds3lbl,"
        "#wdt3:checked~#wdhdr .wds2lbl,"
        "#wdt3:checked~#wdhdr .wds3lbl,"
        "#wdt3:checked~#wdhdr .wds4lbl"
        "{color:#756b63!important;text-decoration:none!important}"
        # Back nav labels
        "#wdt1:checked~#wdnav #wdb1,"
        "#wdt2:checked~#wdnav #wdb2,"
        "#wdt3:checked~#wdnav #wdb3{display:inline-block!important}"
        # Next nav labels
        "#wdt0:checked~#wdnav #wdn0,"
        "#wdt1:checked~#wdnav #wdn1,"
        "#wdt2:checked~#wdnav #wdn2{display:inline-block!important}"
        # Done indicator
        "#wdt3:checked~#wdnav #wdndone{display:inline-block!important}"
        # Hide nav + footer on Detect Dish screen — confirm buttons are the CTA
        "#wdt0:checked~#wdnav{display:none!important}"
        "#wdt0:checked~#wdfooter{display:none!important}"
        # Step counters
        "#wdt0:checked~#wdnav #wdc0,"
        "#wdt1:checked~#wdnav #wdc1,"
        "#wdt2:checked~#wdnav #wdc2,"
        "#wdt3:checked~#wdnav #wdc3{display:block!important}"
        # Info panel toggles in screens (checkbox trick)
        "#wdinf2panel{display:none}"
        "#wdinf2:checked~#wdinf2panel{display:block}"
        "</style>"
    )

    # ── 5-step bar builder ────────────────────────────────────────────────────
    _sep = (
        '<div style="flex:1;height:1px;background:rgba(64,42,31,0.18);'
        'min-width:24px;margin-bottom:22px;align-self:center"></div>'
    )

    def _step(num, label, circ_cls, lbl_cls, for_id=None, locked=False):
        wrap_style = "opacity:0.38;pointer-events:none;" if locked else ""
        if for_id and not locked:
            circ_el = (
                f'<label for="{for_id}" class="{circ_cls}"'
                f' style="width:48px;height:48px;border-radius:999px;display:grid;'
                f'place-items:center;font-size:17px;font-weight:900;cursor:pointer;'
                f'background:rgba(255,255,255,0.58);color:#9e9188;'
                f'border:1px solid rgba(200,190,180,0.5);box-sizing:border-box">{num}</label>'
            )
        else:
            circ_el = (
                f'<div class="{circ_cls}"'
                f' style="width:48px;height:48px;border-radius:999px;display:grid;'
                f'place-items:center;font-size:17px;font-weight:900;'
                f'background:rgba(255,255,255,0.58);color:#9e9188;'
                f'border:1px solid rgba(200,190,180,0.5);box-sizing:border-box">{num}</div>'
            )
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'gap:8px;{wrap_style}">'
            f'{circ_el}'
            f'<div class="{lbl_cls}"'
            f' style="font-size:12px;font-weight:800;color:#9e9188;white-space:nowrap">'
            f'{label}</div>'
            f'</div>'
        )

    # Step 1 (Upload) — always shown as completed
    _step1_done = (
        '<div style="display:flex;flex-direction:column;align-items:center;gap:8px">'
        '<div style="width:48px;height:48px;border-radius:999px;display:grid;'
        'place-items:center;font-size:16px;font-weight:900;'
        'background:rgba(255,255,255,0.82);color:#756b63;'
        'border:1px solid rgba(64,42,31,0.16)">&#10003;</div>'
        '<div style="font-size:12px;font-weight:800;color:#756b63;'
        'white-space:nowrap">Upload</div>'
        '</div>'
    )

    steps_html = (
        _step1_done + _sep
        + _step("2", "Detect Dish",    "wds2circ", "wds2lbl", "wdt0") + _sep
        + _step("3", "Fingerprint",    "wds3circ", "wds3lbl", "wdt1", unlocked < 1) + _sep
        + _step("4", "Wine Pairings",  "wds4circ", "wds4lbl", "wdt2", unlocked < 2) + _sep
        + _step("5", "The Story",       "wds5circ", "wds5lbl", "wdt3", not bool(s4))
    )

    # ── Nav button styles ─────────────────────────────────────────────────────
    _btn_back = (
        "display:none;padding:10px 20px;border-radius:999px;font-size:13px;"
        "border:1px solid rgba(64,42,31,0.18);background:rgba(255,255,255,0.55);"
        "color:#42101d;cursor:pointer;font-weight:700"
    )
    _btn_next = (
        "display:none;padding:11px 26px;border-radius:999px;font-size:13px;"
        "border:none;background:linear-gradient(135deg,#8d1f3a,#5a1024);"
        "color:#fff;cursor:pointer;font-weight:900;"
        "box-shadow:0 10px 24px rgba(122,24,48,0.22)"
    )
    _ctr = (
        "display:none;font-size:11px;color:#756b63;text-transform:uppercase;"
        "letter-spacing:0.12em;font-weight:800;text-align:center"
    )
    n1_lock = f";{_lock}" if unlocked < 1 else ""
    n2_lock = f";{_lock}" if unlocked < 2 else ""
    n3_lock = f";{_lock}" if not s4 else ""

    return f"""<div id="wdshell" style="font-family:'Segoe UI',system-ui,Arial,sans-serif">
{css}
<input type="radio" id="wdt0" name="wdtab"{c[0]} style="display:none">
<input type="radio" id="wdt1" name="wdtab"{c[1]} style="display:none">
<input type="radio" id="wdt2" name="wdtab"{c[2]} style="display:none">
<input type="radio" id="wdt3" name="wdtab"{c[3]} style="display:none">

<!-- ── header ─────────────────────────────────────────────────────── -->
<div id="wdhdr" style="display:grid;grid-template-columns:auto 1fr auto;
     align-items:start;gap:24px;padding:24px 34px 18px;
     border-bottom:1px solid rgba(63,43,35,0.09)">
  <div>
    <div style="font-family:Georgia,'Times New Roman',serif;font-size:30px;
                letter-spacing:-0.8px;font-weight:700;white-space:nowrap;color:#211917">
      Wine<span style="color:#7a1830">&amp;</span>Dine</div>
    <div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
                color:#42101d;font-weight:800;margin-top:6px">Food &amp; wine pairing</div>
  </div>
  <div style="display:flex;align-items:flex-start;justify-content:center;gap:0;width:100%">
    {steps_html}
  </div>
  <div style="width:140px;display:flex;align-items:center;justify-content:center">
    <div style="background:linear-gradient(135deg,#c0334d 0%,#8d1f3a 100%);color:#fff;
               border-radius:999px;font-size:13px;font-weight:800;
               padding:10px 20px;white-space:nowrap;user-select:none;
               box-shadow:0 8px 22px rgba(122,24,48,0.24);text-align:center">
      &#x21BA;&nbsp;&nbsp;Start Over
    </div>
  </div>
</div>

<!-- ── screens — DIRECT children of #wdshell (CSS sibling selector requirement) -->
<div id="wds0" style="padding:28px 34px 20px">{s1 or ""}</div>
<div id="wds1" style="padding:28px 34px 20px">{s2 or _LOADING_SPINNER}</div>
<div id="wds2" style="padding:28px 34px 20px">{s3 or ""}</div>
<div id="wds3" style="padding:28px 34px 20px">{s4 or ""}</div>

<!-- ── nav bar ──────────────────────────────────────────────────────── -->
<div id="wdnav" style="display:flex;justify-content:space-between;align-items:center;
     padding:16px 34px;border-top:1px solid rgba(63,43,35,0.09)">
  <div>
    <label for="wdt0" id="wdb1" style="{_btn_back}">&larr; Back</label>
    <label for="wdt1" id="wdb2" style="{_btn_back}">&larr; Back</label>
    <label for="wdt2" id="wdb3" style="{_btn_back}">&larr; Back</label>
  </div>
  <div>
    <div id="wdc0" style="{_ctr}">Step 2 of 5</div>
    <div id="wdc1" style="{_ctr}">Step 3 of 5</div>
    <div id="wdc2" style="{_ctr}">Step 4 of 5</div>
    <div id="wdc3" style="{_ctr}">Step 5 of 5</div>
  </div>
  <div>
    <label for="wdt1" id="wdn0" style="{_btn_next}{n1_lock}">Next &rarr;</label>
    <label for="wdt2" id="wdn1" style="{_btn_next}{n2_lock}">Next &rarr;</label>
    <label for="wdt3" id="wdn2" style="{_btn_next}{n3_lock}">The Story &rarr;</label>
    <label for="wdt0" id="wdndone" style="{_btn_next}">Done &#10003;</label>
  </div>
</div>

<div id="wdfooter" style="padding:10px 34px 14px;font-size:10px;color:#b8aaa0;text-align:right;
            border-top:1px solid rgba(63,43,35,0.06)">
  Wine&amp;Dine &middot; RSU Advanced ML &middot; 2026
</div>
</div>"""


_LOADING_SPINNER = (
    '<div style="padding:32px;text-align:center;color:#9a9895;font-size:13px">'
    '⏳ Computing…</div>'
)


def _wine_card_parts(food_name: str, conf: float, top5: list,
                     cluster_idx: int, cluster_name: str,
                     sims, desc: str, attn_w):
    """
    Generator — yields the full shell HTML, progressively revealing screens.

    Yield 0: screen1 shown, screen2 loading spinner, cur=1  (BiLSTM running)
    Yield 1: screen2 complete, screen3 loading spinner, cur=1 (wines loading)
    Yield 2: screen3 complete, cur=2  (all done — user auto-lands on screen 3)
    """
    food_key = food_name.lower().replace(" ", "_")
    display  = food_name.replace("_", " ").title() if "_" in food_name else food_name

    s1 = _screen1_html(food_name, conf, top5)

    # ── Yield 0: screen2 is loading ─────────────────────────────────────────
    yield _shell_html(s1, _LOADING_SPINNER, "", 1)

    s2 = _screen2_html(display, desc, attn_w, cluster_idx, cluster_name, sims, "")

    # ── Yield 1: screen2 done, screen3 loading ───────────────────────────────
    yield _shell_html(s1, s2, _LOADING_SPINNER, 1)

    recs = RESULTS_ALL.get(food_key, [])
    safe_cluster = recs[0].get("name", cluster_name) if recs else cluster_name
    feel = _food_feel(safe_cluster)
    s3 = _screen3_html(display, cluster_name, recs, feel)

    s4 = _screen4_html()

    # ── Yield 2: all done, advance to screen3 ───────────────────────────────
    yield _shell_html(s1, s2, s3, 2, s4)



# ── App state ─────────────────────────────────────────────────────────────────
_state: dict = {"food": "", "conf": 0.0, "top5": [], "img_b64": ""}

# ── Event handlers ─────────────────────────────────────────────────────────────
def on_identify(pil_img):
    """CNN pass — populates detect-dish section and shows yes/no confirm row."""
    _nil = (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=True),  gr.update(), gr.update(),
        gr.update(visible=False), gr.update(visible=False),
        gr.update(), gr.update(), gr.update(),
        gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False),
    )
    if pil_img is None:
        return _nil

    food_name, conf, top5 = identify_food(pil_img)
    display = food_name.replace("_", " ").title() if "_" in food_name else food_name

    thumb = pil_img.convert("RGB")
    thumb.thumbnail((480, 480))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=82)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    _state.update(food=food_name, conf=conf, top5=top5, img_b64=img_b64)

    return (
        gr.update(visible=True),                                    # result_col
        gr.update(visible=True),                                    # detect_col
        gr.update(visible=True),                                    # confirm_row
        gr.update(visible=False),                                   # upload_col
        gr.update(value=f"✓  Yes, it's {display}"),                # yes_btn
        gr.update(value="✗  No, correct dish"),                    # no_btn
        gr.update(value=_INFO_CARD_HTML, visible=True),            # info_card
        gr.update(visible=False),                                   # manual_row
        gr.update(value=_s1_head_html(food_name)),                 # screen1_head
        gr.update(value=_s1_photo_html(img_b64)),                  # screen1_photo
        gr.update(value=_s1_info_html(food_name, conf, top5)),     # screen1_info
        gr.update(visible=True),                                    # start_over_btn
        gr.update(visible=False),                                   # info_card_3
        gr.update(visible=False),                                   # info_card_4
    )


def on_yes():
    """BiLSTM pass — 2 yields: spinner → fully ready."""
    food_name = _state.get("food", "")
    food_key  = food_name.lower().replace(" ", "_")
    conf      = _state["conf"]
    top5      = _state["top5"]
    display   = food_name.replace("_", " ").title() if "_" in food_name else food_name

    s1 = _screen1_html(food_name, conf, top5, _state.get("img_b64", ""))

    yield (
        gr.update(visible=False), gr.update(visible=False),
        _shell_html(s1, _LOADING_SPINNER, "", 1, ""),
        gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False),
    )

    cluster_idx, cluster_name, sims, desc, attn_w = bilstm_encode(food_key)
    img_b64 = _state.get("img_b64", "")
    s2 = _screen2_html(display, desc, attn_w, cluster_idx, cluster_name, sims, img_b64)

    recs         = RESULTS_ALL.get(food_key, [])
    safe_cluster = recs[0].get("name", cluster_name) if recs else cluster_name
    feel         = _food_feel(safe_cluster)
    s3           = _screen3_html(display, cluster_name, recs, feel)
    s4           = _screen4_html()

    yield (
        gr.update(visible=False), gr.update(visible=False),
        _shell_html(s1, s2, s3, 1, s4),
        gr.update(visible=False),
        gr.update(visible=True), gr.update(visible=True),
    )


def on_no():
    """Hide Yes/No confirm row, show manual dish input row. Info card stays visible."""
    return gr.update(visible=False), gr.update(visible=True), gr.update()


def on_start_over():
    """Reset all state and return to the upload screen."""
    _state.update(food="", conf=0.0, top5=[], img_b64="")
    return (
        gr.update(visible=True),   # upload_col
        gr.update(visible=False),  # result_col
        gr.update(visible=False),  # info_card
        gr.update(visible=False),  # start_over_btn
        gr.update(visible=False),  # info_card_3
        gr.update(visible=False),  # info_card_4
    )


def on_confirm_dish(dish_text: str):
    """Run the full BiLSTM + wine pipeline from a manually entered dish name."""
    raw = dish_text.strip()
    if not raw:
        yield (
            gr.update(visible=True), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(),
        )
        return

    food_key = raw.lower().replace(" ", "_")
    display  = raw.title()

    img_b64 = _state.get("img_b64", "")
    top5    = _state.get("top5", [])
    conf    = _state.get("conf", 0.0)
    s1      = _screen1_html(food_key, conf, top5, img_b64) if img_b64 else \
              _screen1_html(food_key, 0.0, [(food_key, 1.0)], "")
    yield (
        gr.update(visible=False), gr.update(visible=False),
        _shell_html(s1, _LOADING_SPINNER, "", 1, ""),
        gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False),
    )

    cluster_idx, cluster_name, sims, desc, attn_w = bilstm_encode(food_key)
    s2 = _screen2_html(display, desc, attn_w, cluster_idx, cluster_name, sims, img_b64)

    recs         = RESULTS_ALL.get(food_key, [])
    safe_cluster = recs[0].get("name", cluster_name) if recs else cluster_name
    feel         = _food_feel(safe_cluster)
    s3           = _screen3_html(display, cluster_name, recs, feel)
    s4           = _screen4_html()

    yield (
        gr.update(visible=False), gr.update(visible=False),
        _shell_html(s1, s2, s3, 1, s4),
        gr.update(visible=False),
        gr.update(visible=True), gr.update(visible=True),
    )


# ── Info cards — shown below the glass card, one per screen ───────────────────
def _info_card(label_a: str, value_a: str, desc_a: str,
               label_b: str, value_b: str, desc_b: str) -> str:
    """Standard 'What happens in the background?' expandable card — 2 inner blocks."""
    return f"""
<div style="margin-top:14px;border-radius:16px;overflow:hidden;
            background:linear-gradient(135deg,rgba(122,24,48,0.05),rgba(201,161,93,0.08));
            border:1px solid rgba(122,24,48,0.12);
            font-family:'Segoe UI',system-ui,Arial,sans-serif">
  <details>
    <summary style="display:flex;align-items:center;gap:10px;
                    padding:15px 22px;cursor:pointer;list-style:none;
                    font-size:13px;font-weight:700;color:#42101d">
      What happens in the background?
      <span style="margin-left:auto;font-size:16px;color:#9e9188;font-weight:400;line-height:1">+</span>
    </summary>
    <div style="padding:0 18px 18px;display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div style="background:rgba(255,255,255,0.80);border-radius:14px;padding:18px">
        <div style="font-size:10px;font-weight:800;color:#7a1830;text-transform:uppercase;
                    letter-spacing:0.13em;margin-bottom:8px">{label_a}</div>
        <div style="font-size:15px;color:#211917;font-weight:700;margin-bottom:6px">{value_a}</div>
        <div style="font-size:12px;color:#7c726b;line-height:1.6">{desc_a}</div>
      </div>
      <div style="background:rgba(255,255,255,0.80);border-radius:14px;padding:18px">
        <div style="font-size:10px;font-weight:800;color:#7a1830;text-transform:uppercase;
                    letter-spacing:0.13em;margin-bottom:8px">{label_b}</div>
        <div style="font-size:15px;color:#211917;font-weight:700;margin-bottom:6px">{value_b}</div>
        <div style="font-size:12px;color:#7c726b;line-height:1.6">{desc_b}</div>
      </div>
    </div>
  </details>
</div>
"""

_INFO_CARD_HTML = _info_card(
    "MODEL", "ResNet-50",
    "A deep learning model trained to recognise 101 different dishes from a single photo",
    "OUTPUT", "Probability scores",
    "The most probable dish detected in your photo, and other runner ups",
)

_INFO_CARD_3_HTML = _info_card(
    "MODEL", "TasteBiLSTM + Bahdanau attention",
    "A bidirectional LSTM that reads the flavor description and weights each word by how taste-defining it is",
    "OUTPUT", "Flavor cluster",
    "Cosine similarity to 9 pre-learned flavor profile centroids picks the closest match",
)

_INFO_CARD_4_HTML = _info_card(
    "METHOD", "Tiered wine matching",
    "Three objectives pick three wines: nearest flavor match, surprising angle, deliberate contrast",
    "OUTPUT", "3 wine pairings",
    "Safe Bet (cluster centroid), Hidden Gem (alt pairing), Bold Move (max contrast)",
)

# ── Full-screen app CSS ────────────────────────────────────────────
_APP_CSS = """
html, body {
  margin: 0 !important; padding: 0 !important;
  background:
    radial-gradient(circle at 12% 6%, rgba(255,255,255,0.90), transparent 28%),
    radial-gradient(circle at 18% 10%, rgba(198,112,103,0.13), transparent 34%),
    radial-gradient(circle at 86% 12%, rgba(122,24,48,0.14), transparent 32%),
    radial-gradient(circle at 70% 88%, rgba(201,161,93,0.16), transparent 36%),
    linear-gradient(135deg, #fcf8f2 0%, #f3ebe2 46%, #efe3da 100%) !important;
  min-height: 100vh;
}
.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: 32px 16px !important;
  background: transparent !important;
  min-height: 100vh;
  width: 100% !important;
}
footer { display: none !important; }
div.main { padding: 0 !important; background: transparent !important; }

/* Default: strip Gradio's borders/backgrounds/padding from every block, but
   DO NOT force width:100% globally (that would break nested scaled columns) */
.gradio-container .block,
.gradio-container .form {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  padding: 0 !important;
  margin: 0 !important;
}
.gap { gap: 0 !important; }

/* TOP-LEVEL containers — every screen card and every info card MUST be exactly
   the same width = full content width of .gradio-container. No Gradio default
   max-width or auto-sizing allowed here. */
#wdupload,
#wdcard_outer,
#wdinfo,
#wdinfo3,
#wdinfo4 {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  margin: 0 !important;
  box-sizing: border-box !important;
  flex: 0 0 auto !important;
}

/* Upload column = a single glass card (same styling as #wdcard_outer below) */
#wdupload {
  background: rgba(255,250,244,0.72) !important;
  backdrop-filter: blur(28px) saturate(1.08) !important;
  border: 1px solid rgba(255,255,255,0.72) !important;
  border-radius: 34px !important;
  box-shadow: 0 34px 90px rgba(52,34,26,0.10),
              inset 0 1px 0 rgba(255,255,255,0.75) !important;
  overflow: hidden !important;
}
#wdupload > .wrap {
  padding: 0 !important;
  gap: 0 !important;
}
/* Image upload zone — sits inside the same glass card, no separate card look */
#wdfood {
  margin: 0 !important;
  padding: 0 34px !important;
}
#wdfood > .wrap {
  border-radius: 18px !important;
  overflow: hidden !important;
  padding: 0 !important;
}
#wdfood .upload-container, #wdfood [data-testid="image"] {
  border: 1.5px dashed rgba(122,24,48,0.26) !important;
  border-radius: 18px !important;
  background: rgba(255,250,244,0.45) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.4) !important;
  min-height: 220px !important;
}
#wdfood .upload-container:hover,
#wdfood [data-testid="image"]:hover {
  border-color: rgba(122,24,48,0.42) !important;
}
/* Hide Gradio's secondary source buttons (clipboard icon strip) */
#wdfood .source-selection,
#wdfood [data-testid="source-select"] {
  display: none !important;
}
/* Analyze button */
#wdanalyze,
#wdanalyze button {
  background: linear-gradient(135deg,#8d1f3a 0%,#5a1024 100%) !important;
  color: #fff !important; border: none !important; border-radius: 999px !important;
  font-size: 15px !important; font-weight: 900 !important;
  padding: 14px 32px !important;
  box-shadow: 0 14px 30px rgba(122,24,48,0.22) !important;
  letter-spacing: -0.2px !important;
  width: 100% !important;
  min-width: 0 !important;
  display: block !important;
}
#wdanalyze:hover,
#wdanalyze button:hover { opacity: 0.88 !important; }
/* Confirm buttons */
#wdyes,
#wdyes button {
  background: linear-gradient(135deg,#8d1f3a 0%,#5a1024 100%) !important;
  color: #fff !important; border: none !important; border-radius: 999px !important;
  font-weight: 900 !important; padding: 13px 28px !important;
  box-shadow: 0 10px 24px rgba(122,24,48,0.22) !important;
}
#wdyes:hover,
#wdyes button:hover,
#wdanalyze:hover,
#wdanalyze button:hover {
  filter: brightness(0.96) saturate(1.02) !important;
  opacity: 1 !important;
}
#wdno,
#wdno button {
  background: rgba(122,24,48,0.07) !important; color: #5a1024 !important;
  border: 1px solid rgba(122,24,48,0.34) !important; border-radius: 999px !important;
  font-weight: 800 !important; padding: 13px 24px !important;
}

/* Upload CTA — sits inside the same glass card with matching horizontal padding */
#wduploadcta {
  padding: 16px 34px 28px !important;
}
#wduploadcta > .wrap {
  padding: 0 !important;
  width: 100% !important;
  flex-direction: column !important;
}
#wdanalyze {
  width: 100% !important;
  min-width: 0 !important;
}
/* Result card — glass wrapper (Gradio Column) */
#wdcard_outer {
  background: rgba(255,250,244,0.72) !important;
  backdrop-filter: blur(28px) saturate(1.08) !important;
  border: 1px solid rgba(255,255,255,0.72) !important;
  border-radius: 34px !important;
  box-shadow: 0 34px 90px rgba(52,34,26,0.10),
              inset 0 1px 0 rgba(255,255,255,0.75) !important;
  overflow: hidden !important;
  position: relative !important;
}
#wdcard_outer > .wrap {
  padding: 0 !important;
  gap: 0 !important;
}
#wdcard > .wrap { padding: 0 !important; background: transparent !important; }

/* Screen 2: detect-dish layout inside the glass card */
#wddetect {
  padding: 0 !important;
}
#wddetect > .wrap {
  padding: 0 !important;
  gap: 0 !important;
}
#wddetect_hdr {
  padding: 0 !important;
}
#wddetect_body {
  padding: 32px 34px 28px !important;
}
#wddetect_body > .wrap {
  padding: 0 !important;
  gap: 0 !important;
}
#wds1body > .wrap {
  padding: 0 !important;
  gap: 28px !important;
  align-items: start !important;
}
#wds1photocol > .wrap,
#wds1right > .wrap {
  padding: 0 !important;
  gap: 0 !important;
}

/* Confirm and manual — stacked vertically inside right column, no spacer needed */
#wdconfirm > .wrap,
#wdmanual > .wrap {
  padding: 16px 0 0 !important;
  gap: 10px !important;
  flex-direction: column !important;
}
#wdconfirm > .wrap > *,
#wdmanual > .wrap > * { width: 100% !important; }

/* Manual dish input textbox — wine-tinted, prominent red border */
#wddishinput textarea {
  border-radius: 12px !important;
  border: 1.5px solid rgba(122,24,48,0.42) !important;
  font-size: 14px !important;
  padding: 12px 16px !important;
  background: rgba(255,248,245,0.95) !important;
  color: #42101d !important;
  resize: none !important;
}
#wddishinput textarea::placeholder {
  color: rgba(122,24,48,0.40) !important;
}
#wddishinput textarea:focus {
  border-color: rgba(122,24,48,0.70) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(122,24,48,0.14) !important;
  background: rgba(255,250,247,1) !important;
}
/* Start Over — transparent click-catcher overlaying the visual button in the header.
   opacity:0 hides it; position:absolute places it exactly over the 140px spacer column.
   No JS needed: real Gradio button click fires on_start_over directly. */
#wdstartover {
  position: absolute !important;
  top: 24px !important;
  right: 34px !important;
  width: 140px !important;
  height: 48px !important;
  opacity: 0 !important;
  z-index: 200 !important;
  cursor: pointer !important;
  margin: 0 !important;
  padding: 0 !important;
}
#wdstartover button,
#wdstartover button.primary,
#wdstartover button.secondary {
  width: 100% !important;
  height: 100% !important;
  cursor: pointer !important;
  padding: 0 !important;
  margin: 0 !important;
  border: none !important;
  border-radius: 0 !important;
}
/* Confirm dish button — wine-red (bulletproof selectors override Gradio primary blue) */
#wdconfirmdish,
#wdconfirmdish button,
#wdconfirmdish button.primary,
button#wdconfirmdish {
  background: linear-gradient(135deg,#8d1f3a,#5a1024) !important;
  border: none !important; color: #fff !important;
  border-radius: 999px !important;
  font-size: 14px !important; font-weight: 800 !important;
  box-shadow: 0 8px 20px rgba(122,24,48,0.20) !important;
}
#wdconfirmdish button:hover {
  background: linear-gradient(135deg,#a0243f,#6e1430) !important;
}
/* Carousel info cards — show only the one that matches the active screen */
#wdinfo3, #wdinfo4 {
  display: none !important;
}
body:has(#wdt1:checked) #wdinfo3 {
  display: block !important;
}
body:has(#wdt2:checked) #wdinfo4 {
  display: block !important;
}
#wdinfo3, #wdinfo4 {
  margin-top: 14px !important;
}
/* ── Welcome screen — landing splash before the upload step ──────────────── */
/* NOTE: do NOT set `display:flex !important` here — it would override Gradio's
   `display:none` when visible=False, leaving the welcome card stuck on screen. */
#wdwelcome {
  background: rgba(255,250,244,0.72) !important;
  backdrop-filter: blur(28px) saturate(1.08) !important;
  border: 1px solid rgba(255,255,255,0.72) !important;
  border-radius: 34px !important;
  box-shadow: 0 34px 90px rgba(52,34,26,0.10),
              inset 0 1px 0 rgba(255,255,255,0.75) !important;
  padding: 96px 48px 88px !important;
  text-align: center !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  margin: 0 !important;
  box-sizing: border-box !important;
}
#wdwelcome > .wrap {
  padding: 0 !important;
  gap: 0 !important;
  align-items: center !important;
}
#wdstart,
#wdstart button,
#wdstart button.primary,
button#wdstart {
  background: linear-gradient(135deg,#8d1f3a 0%,#5a1024 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 999px !important;
  font-size: 15px !important;
  font-weight: 900 !important;
  letter-spacing: 0.04em !important;
  padding: 16px 40px !important;
  box-shadow: 0 14px 36px rgba(122,24,48,0.26) !important;
  cursor: pointer !important;
  width: auto !important;
  min-width: 220px !important;
  white-space: nowrap !important;
}
#wdstart:hover,
#wdstart button:hover {
  background: linear-gradient(135deg,#a0243f 0%,#6e1430 100%) !important;
  filter: brightness(1.02) !important;
}
"""
_UPLOAD_HEADER_HTML = """
<div style="font-family:'Segoe UI',system-ui,Arial,sans-serif">

  <!-- header row: brand | step bar | spacer -->
  <div style="display:grid;grid-template-columns:auto 1fr auto;
              align-items:start;gap:24px;padding:24px 34px 18px;
              border-bottom:1px solid rgba(63,43,35,0.09)">
    <div>
      <div style="font-family:Georgia,'Times New Roman',serif;font-size:30px;
                  letter-spacing:-0.8px;font-weight:700;white-space:nowrap;color:#211917">
        Wine<span style="color:#7a1830">&amp;</span>Dine</div>
      <div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
                  color:#42101d;font-weight:800;margin-top:6px">Food &amp; wine pairing</div>
    </div>
    <!-- 5-step indicator -->
    <div style="display:flex;align-items:center;justify-content:center;gap:0;width:100%">
      <!-- step 1 active -->
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div style="width:48px;height:48px;border-radius:999px;display:grid;
                    place-items:center;background:linear-gradient(135deg,#8d1f3a,#5a1024);
                    color:#fff;font-size:17px;font-weight:900;
                    box-shadow:0 10px 28px rgba(122,24,48,0.24)">1</div>
        <div style="font-size:12px;font-weight:800;color:#42101d;
                    text-decoration:underline;text-underline-offset:4px;white-space:nowrap">Upload</div>
      </div>
      <div style="flex:1;min-width:24px;height:1px;background:rgba(64,42,31,0.18);margin-bottom:22px"></div>
      <!-- step 2 -->
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div style="width:48px;height:48px;border-radius:999px;display:grid;
                    place-items:center;background:rgba(255,255,255,0.58);
                    color:#9e9188;font-size:17px;font-weight:900;
                    border:1px solid rgba(200,190,180,0.5)">2</div>
        <div style="font-size:12px;font-weight:800;color:#9e9188;white-space:nowrap">Detect Dish</div>
      </div>
      <div style="flex:1;min-width:24px;height:1px;background:rgba(64,42,31,0.18);margin-bottom:22px"></div>
      <!-- step 3 -->
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div style="width:48px;height:48px;border-radius:999px;display:grid;
                    place-items:center;background:rgba(255,255,255,0.58);
                    color:#9e9188;font-size:17px;font-weight:900;
                    border:1px solid rgba(200,190,180,0.5)">3</div>
        <div style="font-size:12px;font-weight:800;color:#9e9188;white-space:nowrap">Fingerprint</div>
      </div>
      <div style="flex:1;min-width:24px;height:1px;background:rgba(64,42,31,0.18);margin-bottom:22px"></div>
      <!-- step 4 -->
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div style="width:48px;height:48px;border-radius:999px;display:grid;
                    place-items:center;background:rgba(255,255,255,0.58);
                    color:#9e9188;font-size:17px;font-weight:900;
                    border:1px solid rgba(200,190,180,0.5)">4</div>
        <div style="font-size:12px;font-weight:800;color:#9e9188;white-space:nowrap">Wine Pairings</div>
      </div>
      <div style="flex:1;min-width:24px;height:1px;background:rgba(64,42,31,0.18);margin-bottom:22px"></div>
      <!-- step 5 -->
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div style="width:48px;height:48px;border-radius:999px;display:grid;
                    place-items:center;background:rgba(255,255,255,0.58);
                    color:#9e9188;font-size:17px;font-weight:900;
                    border:1px solid rgba(200,190,180,0.5)">5</div>
        <div style="font-size:12px;font-weight:800;color:#9e9188;white-space:nowrap">The Story</div>
      </div>
    </div>
    <div style="width:140px"></div>
  </div>

  <!-- intro copy -->
  <div style="padding:34px 34px 28px">
    <div style="font-size:12px;color:#7a1830;text-transform:uppercase;
                letter-spacing:0.14em;font-weight:800;margin-bottom:10px">STEP 1 &middot; UPLOAD</div>
    <div style="font-family:Georgia,'Times New Roman',serif;font-size:40px;
                font-weight:700;color:#211917;letter-spacing:-1.4px;line-height:1;
                margin-bottom:8px">What are you eating?</div>
    <div style="font-size:13px;color:#9e9188;line-height:1.6;max-width:560px">
      Upload a food photo and the app will identify the dish,
      build a taste fingerprint and suggest wines that fit the experience.
    </div>
  </div>
</div>
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(),
    title="Wine & Dine 🍷",
    css=_APP_CSS,
) as demo:

    # ── Welcome / landing screen ──────────────────────────────────────────────
    with gr.Column(visible=True, elem_id="wdwelcome") as welcome_col:
        gr.HTML("""
<div style="font-family:'Segoe UI',system-ui,Arial,sans-serif;
            display:flex;flex-direction:column;align-items:center;gap:0;width:100%">
  <div style="font-size:11px;color:#7a1830;text-transform:uppercase;
              letter-spacing:0.28em;font-weight:800;margin-bottom:22px">
    RSU &middot; Advanced Machine Learning
  </div>
  <div style="font-family:Georgia,'Times New Roman',serif;font-size:78px;
              font-weight:700;line-height:1;letter-spacing:-2.2px;
              color:#211917;margin-bottom:16px">
    Wine<span style="color:#7a1830">&amp;</span>Dine
  </div>
  <div style="font-size:14px;color:#7a1830;text-transform:uppercase;
              letter-spacing:0.18em;font-weight:800;margin-bottom:36px">
    Food &amp; wine pairing
  </div>
  <div style="font-size:16px;color:#5e544d;line-height:1.65;
              max-width:560px;margin-bottom:44px">
    A multimodal AI system that pairs wines to dishes from a single photo —
    combining visual recognition with the semantic space of taste.
  </div>
</div>
""")
        start_btn = gr.Button(
            "Let's start  →", variant="primary", elem_id="wdstart",
        )

    # ── Upload screen (screen 0) ──────────────────────────────────────────────
    with gr.Column(visible=False, elem_id="wdupload") as upload_col:
        gr.HTML(_UPLOAD_HEADER_HTML)
        img_input = gr.Image(
            type="pil", label="", show_label=False,
            height=240, elem_id="wdfood",
            sources=["upload"],
        )
        with gr.Row(elem_id="wduploadcta"):
            identify_btn = gr.Button(
                "Analyze dish →", variant="primary",
                elem_id="wdanalyze",
            )

    # ── Result card (glass card wrapper) ─────────────────────────────────────
    with gr.Column(visible=False, elem_id="wdcard_outer") as result_col:

        # Detect Dish section — shell header + photo left, info+buttons right
        with gr.Column(visible=False, elem_id="wddetect") as detect_col:
            gr.HTML(value=_DETECT_HEADER_HTML, elem_id="wddetect_hdr")
            with gr.Column(elem_id="wddetect_body"):
                screen1_head  = gr.HTML(value="", elem_id="wds1head")
                with gr.Row(elem_id="wds1body"):
                    with gr.Column(scale=11, elem_id="wds1photocol"):
                        screen1_photo = gr.HTML(value="", elem_id="wds1photo")
                    with gr.Column(scale=9, elem_id="wds1right"):
                        screen1_info = gr.HTML(value="", elem_id="wds1info")
                        with gr.Column(visible=False, elem_id="wdconfirm") as confirm_row:
                            yes_btn = gr.Button(
                                "✓  Yes, it's my dish",
                                variant="primary", elem_id="wdyes",
                            )
                            no_btn = gr.Button(
                                "✗  No, correct dish",
                                variant="secondary", elem_id="wdno",
                            )
                        with gr.Column(visible=False, elem_id="wdmanual") as manual_row:
                            dish_input = gr.Textbox(
                                placeholder="e.g. pasta, sushi, burger…",
                                label="", show_label=False,
                                lines=1, max_lines=1,
                                elem_id="wddishinput",
                            )
                            confirm_dish_btn = gr.Button(
                                "Confirm dish →",
                                variant="primary", elem_id="wdconfirmdish",
                            )

        # Screens 2/3/4 carousel (shown after Yes is confirmed)
        wine_card = gr.HTML(value="", elem_id="wdcard")
        start_over_btn = gr.Button("↺  Start Over", elem_id="wdstartover", visible=False)

    # ── Info cards — outside the glass card, one per screen state ────────────
    info_card   = gr.HTML(value="",                 elem_id="wdinfo",  visible=False)
    info_card_3 = gr.HTML(value=_INFO_CARD_3_HTML, elem_id="wdinfo3", visible=False)
    info_card_4 = gr.HTML(value=_INFO_CARD_4_HTML, elem_id="wdinfo4", visible=False)

    # ── Event wiring ──────────────────────────────────────────────────────────
    start_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=None,
        outputs=[welcome_col, upload_col],
    )
    identify_btn.click(
        on_identify,
        inputs=[img_input],
        outputs=[result_col, detect_col, confirm_row, upload_col,
                 yes_btn, no_btn, info_card, manual_row,
                 screen1_head, screen1_photo, screen1_info, start_over_btn,
                 info_card_3, info_card_4],
    )
    yes_btn.click(
        on_yes,
        inputs=None,
        outputs=[confirm_row, detect_col, wine_card, info_card,
                 info_card_3, info_card_4],
    )
    no_btn.click(
        on_no,
        inputs=None,
        outputs=[confirm_row, manual_row, info_card],
    )
    confirm_dish_btn.click(
        on_confirm_dish,
        inputs=[dish_input],
        outputs=[manual_row, detect_col, wine_card, info_card,
                 info_card_3, info_card_4],
    )
    start_over_btn.click(
        on_start_over,
        inputs=None,
        outputs=[upload_col, result_col, info_card, start_over_btn,
                 info_card_3, info_card_4],
    )

if __name__ == "__main__":
    demo.launch()
