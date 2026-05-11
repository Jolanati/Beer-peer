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
            f' style="width:100%;height:100%;object-fit:cover;display:block;'
            f'border-radius:20px">'
        )
    else:
        photo_el = (
            '<div style="width:100%;height:360px;display:flex;align-items:center;'
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

      <div style="margin-top:10px;font-size:12px;color:#b8aaa0;line-height:1.6">
        Use the buttons below to confirm or pick a different dish.</div>
    </div>
  </div>

  <!-- "what happens in background" details -->
  <details style="margin-top:22px;border:1px solid rgba(63,43,35,0.09);
                  border-radius:14px;overflow:hidden">
    <summary style="padding:12px 18px;font-size:12px;font-weight:700;
                    color:#7c726b;cursor:pointer;list-style:none;
                    background:rgba(251,247,241,0.8)">
      &#9432; What happens in the background?</summary>
    <div style="padding:14px 18px;font-size:12px;color:#7c726b;
                line-height:1.75;background:rgba(251,247,241,0.5)">
      A <strong style="color:#211917">ResNet-50</strong> trained on Food-101
      (101 classes, ~75k images) predicts the food category from your photo.
      The model outputs a probability distribution over all 101 classes &mdash;
      the top prediction and its confidence score are shown above.
      Only the top class is passed to the next step.
    </div>
  </details>

</div>"""


def _screen2_html(display: str, desc: str, attn_w,
                  cluster_idx: int, cluster_name: str, sims,
                  img_b64: str = "") -> str:
    """Screen 2 (step 3) — taste fingerprint: photo left, heatmap + bars right."""

    # ── Attention-highlighted word chips ─────────────────────────────────────
    words    = desc.split()[:MAX_SEQ_LEN]
    attn_arr = attn_w[:len(words)]
    a_min, a_max = attn_arr.min(), attn_arr.max()
    attn_norm = (attn_arr - a_min) / (a_max - a_min + 1e-8)
    word_html = ""
    for w_txt, a in zip(words, attn_norm):
        if a >= 0.75:
            chip = ("background:#7a1830;color:#fff;font-weight:800;"
                    "border:1px solid transparent")
        elif a >= 0.40:
            chip = ("background:rgba(201,161,93,0.28);color:#42101d;"
                    "font-weight:600;border:1px solid rgba(201,161,93,0.4)")
        elif a >= 0.15:
            chip = ("background:rgba(122,24,48,0.07);color:#42101d;"
                    "border:1px solid rgba(122,24,48,0.12)")
        else:
            chip = "background:transparent;color:#7c726b;border:1px solid transparent"
        word_html += (
            f'<span style="display:inline-block;border-radius:6px;'
            f'padding:3px 8px;margin:3px 2px;font-size:13px;'
            f'line-height:1.5;cursor:default;{chip}">{w_txt}</span>'
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

  <!-- step label + heading -->
  <div style="font-size:11px;color:#7a1830;text-transform:uppercase;
              letter-spacing:0.14em;font-weight:800;margin-bottom:10px">
    STEP 3 &middot; TASTE FINGERPRINT</div>
  <div style="font-family:Georgia,'Times New Roman',serif;font-size:36px;
              font-weight:700;color:#211917;letter-spacing:-1.2px;
              line-height:1;margin-bottom:6px">How does this {display} taste?</div>
  <div style="font-size:13px;color:#9e9188;margin-bottom:24px">
    We turn the verified dish into a flavor profile the wine matcher can use.</div>

  <!-- 2-col layout: photo LEFT | heatmap + bars RIGHT -->
  <div style="display:grid;grid-template-columns:0.9fr 1.1fr;gap:28px;
              align-items:start;margin-bottom:24px">

    <!-- LEFT: food photo -->
    <div style="border-radius:16px;overflow:hidden;
                box-shadow:0 12px 32px rgba(52,34,26,0.12);
                aspect-ratio:4/3">
      {photo_el}
    </div>

    <!-- RIGHT: heatmap + cluster bars -->
    <div style="display:flex;flex-direction:column;gap:22px">

      <!-- taste fingerprint badge -->
      <div style="padding:14px 16px;background:rgba(122,24,48,0.06);
                  border-radius:14px;border-left:3px solid #7a1830">
        <div style="font-size:10px;color:#b8aaa0;text-transform:uppercase;
                    letter-spacing:0.12em;font-weight:800;margin-bottom:4px">
          Flavor profile</div>
        <div style="font-family:Georgia,'Times New Roman',serif;font-size:22px;
                    font-weight:700;color:#42101d;letter-spacing:-0.5px;
                    line-height:1.2">{cluster_name}</div>
      </div>

      <!-- attention heatmap -->
      <div>
        <div style="display:flex;justify-content:space-between;
                    align-items:center;margin-bottom:8px">
          <div style="font-size:11px;color:#9e9188;text-transform:uppercase;
                      letter-spacing:0.1em;font-weight:800">Attention heatmap</div>
          <div style="font-size:10px;color:#b8aaa0;display:flex;align-items:center;gap:6px">
            <span style="display:inline-block;width:9px;height:9px;border-radius:2px;
                         background:#7a1830;vertical-align:middle"></span>high
            <span style="display:inline-block;width:9px;height:9px;border-radius:2px;
                         background:rgba(201,161,93,0.28);vertical-align:middle;
                         margin-left:6px"></span>medium
          </div>
        </div>
        <div style="background:rgba(251,247,241,0.8);border-radius:14px;
                    padding:12px 14px;border:1px solid rgba(63,43,35,0.09);
                    line-height:2;min-height:72px">
          {word_html}
        </div>
        <div style="font-size:10px;color:#b8aaa0;margin-top:5px">
          Word warmth = Bahdanau attention weight &middot; darker = more influential</div>
      </div>

      <!-- cluster similarity bars -->
      <div>
        <div style="font-size:10px;color:#b8aaa0;text-transform:uppercase;
                    letter-spacing:0.12em;font-weight:800;margin-bottom:12px">
          Closest flavor worlds</div>
        {cluster_rows}
      </div>

    </div>
  </div>

  <!-- "what happens in background" details -->
  <details style="margin-bottom:22px;border:1px solid rgba(63,43,35,0.09);
                  border-radius:14px;overflow:hidden">
    <summary style="padding:12px 18px;font-size:12px;font-weight:700;
                    color:#7c726b;cursor:pointer;list-style:none;
                    background:rgba(251,247,241,0.8)">
      + What happens in the background?</summary>
    <div style="padding:14px 18px;font-size:12px;color:#7c726b;
                line-height:1.75;background:rgba(251,247,241,0.5)">
      The flavor description is tokenised and passed through a
      <strong style="color:#211917">bidirectional LSTM</strong> with
      <strong style="color:#211917">Bahdanau attention</strong>
      (hidden=256, 2 layers, dropout=0.4, vocab=4294, max_seq=64).
      The attention-weighted context vector (512-d) is L2-normalised and compared
      by cosine similarity to 9 pre-computed cluster centroids
      (BisectingKMeans, TF-IDF named) — the closest cluster becomes the flavor profile.
    </div>
  </details>

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

  <!-- overline + heading -->
  <div style="text-align:center;margin-bottom:28px">
    <div style="font-size:10px;color:#b8aaa0;text-transform:uppercase;
                letter-spacing:0.18em;font-weight:800;margin-bottom:10px">
      THE STORY BEHIND THE PROJECT</div>
    <div style="font-family:Georgia,'Times New Roman',serif;font-size:38px;
                font-weight:700;color:#211917;letter-spacing:-1.5px;
                line-height:1;margin-bottom:16px">The Story of Wine&amp;Dine</div>
    <div style="font-size:14px;color:#7c726b;line-height:1.75;
                max-width:580px;margin:0 auto">
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
        rating   = rec.get("rating", "—")
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
        for kw in kws[:3]:
            tags_html += (
                f'<span style="font-size:10px;font-weight:600;padding:3px 9px;'
                f'border-radius:20px;background:{tag_bg};color:{color};'
                f'white-space:nowrap">{kw}</span>'
            )

        cards_html += f"""
<div style="border-radius:18px;overflow:hidden;background:#fff;
            border:1px solid rgba(63,43,35,0.09);
            box-shadow:0 4px 16px rgba(52,34,26,0.07)">

  <!-- tier header strip -->
  <div style="background:{strip_bg};padding:10px 14px;
              display:flex;align-items:center;justify-content:space-between">
    <div style="display:flex;align-items:center;gap:6px">
      <span style="font-size:11px;font-weight:900;color:{color}">{icon}</span>
      <span style="font-size:10px;font-weight:800;letter-spacing:0.08em;
                   text-transform:uppercase;color:{color}">{tier_lbl}</span>
    </div>
    <div style="text-align:right">
      <div style="font-family:Georgia,serif;font-size:22px;font-weight:700;
                  color:{color};line-height:1;letter-spacing:-0.5px">{conf_pct}%</div>
      <div style="font-size:9px;color:#b8aaa0;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.08em">{conf_lbl}</div>
    </div>
  </div>

  <!-- confidence bar -->
  <div style="height:3px;background:rgba(64,42,31,0.08)">
    <div style="background:{color};width:{conf_pct}%;height:100%"></div>
  </div>

  <!-- card body -->
  <div style="padding:14px;display:flex;flex-direction:column;gap:9px">

    <!-- wine name + year -->
    <div>
      <div style="font-family:Georgia,'Times New Roman',serif;font-size:18px;
                  font-weight:700;color:#211917;line-height:1.25">{wine_name}</div>
      {year_html}
    </div>

    <!-- reasoning -->
    <div style="font-size:12px;color:#7c726b;line-height:1.6">
      {reasoning}
    </div>

    <!-- taste tags -->
    <div style="display:flex;flex-wrap:wrap;gap:5px">
      {tags_html}
    </div>

    <!-- divider -->
    <div style="height:1px;background:rgba(63,43,35,0.09)"></div>

    <!-- rating -->
    <div style="display:flex;align-items:center;gap:4px">
      <span style="color:#c9a15d;font-size:12px">&#9733;</span>
      <span style="font-size:13px;font-weight:700;color:#42101d">{rating}</span>
      <span style="font-size:10px;color:#b8aaa0">/ 100</span>
    </div>

    <!-- review quote -->
    <div style="font-size:11px;color:#7c726b;font-style:italic;line-height:1.65;
                border-left:2px solid {color};padding-left:10px">
      &ldquo;{snippet}&rdquo;
    </div>

  </div>
</div>"""

    return f"""
<div style="font-family:'Segoe UI',system-ui,Arial,sans-serif">

  <!-- heading row: title left, subtitle right -->
  <div style="display:grid;grid-template-columns:auto 1fr;gap:24px;
              align-items:start;margin-bottom:22px">
    <div>
      <div style="font-family:Georgia,'Times New Roman',serif;font-size:36px;
                  font-weight:700;color:#211917;letter-spacing:-1.2px;line-height:1">
        Wine pairings</div>
    </div>
    <div style="padding-top:10px">
      <div style="font-size:13px;color:#9e9188;line-height:1.5">
        The result screen is the reward: three directions,<br>each with a different pairing logic.</div>
    </div>
  </div>

  <!-- 3-col card grid -->
  <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
              gap:14px;margin-bottom:22px">
    {cards_html}
  </div>

  <!-- "what happens in background" details -->
  <details style="border:1px solid rgba(63,43,35,0.09);
                  border-radius:14px;overflow:hidden">
    <summary style="padding:12px 18px;font-size:12px;font-weight:700;
                    color:#7c726b;cursor:pointer;list-style:none;
                    background:rgba(251,247,241,0.8)">
      + what happens in background?</summary>
    <div style="padding:14px 18px;font-size:12px;color:#7c726b;
                line-height:1.75;background:rgba(251,247,241,0.5)">
      For each tier, a different objective function picks the wine.
      <strong style="color:#211917">Safe Bet</strong>: nearest cluster centroid by
      cosine similarity to the dominant flavor description.
      <strong style="color:#211917">Hidden Gem</strong>: nearest centroid to the
      &lsquo;surprising pairing&rsquo; description.
      <strong style="color:#211917">Bold Move</strong>: maximises contrast while
      keeping the wine drinkable. Within each tier, the final wine is the highest
      cosine-similarity match from a 10-wine pool curated for that cluster.
    </div>
  </details>

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
        'min-width:24px;margin-bottom:20px;align-self:center"></div>'
    )

    def _step(num, label, circ_cls, lbl_cls, for_id=None, locked=False):
        wrap_style = "opacity:0.38;pointer-events:none;" if locked else ""
        if for_id and not locked:
            circ_el = (
                f'<label for="{for_id}" class="{circ_cls}"'
                f' style="width:40px;height:40px;border-radius:999px;display:grid;'
                f'place-items:center;font-size:16px;font-weight:900;cursor:pointer;'
                f'background:rgba(255,255,255,0.58);color:#9e9188;'
                f'border:1px solid rgba(200,190,180,0.5);box-sizing:border-box">{num}</label>'
            )
        else:
            circ_el = (
                f'<div class="{circ_cls}"'
                f' style="width:40px;height:40px;border-radius:999px;display:grid;'
                f'place-items:center;font-size:16px;font-weight:900;'
                f'background:rgba(255,255,255,0.58);color:#9e9188;'
                f'border:1px solid rgba(200,190,180,0.5);box-sizing:border-box">{num}</div>'
            )
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'gap:8px;{wrap_style}">'
            f'{circ_el}'
            f'<div class="{lbl_cls}"'
            f' style="font-size:11px;font-weight:800;color:#9e9188;white-space:nowrap">'
            f'{label}</div>'
            f'</div>'
        )

    # Step 1 (Upload) — always shown as completed
    _step1_done = (
        '<div style="display:flex;flex-direction:column;align-items:center;gap:8px">'
        '<div style="width:40px;height:40px;border-radius:999px;display:grid;'
        'place-items:center;font-size:14px;font-weight:900;'
        'background:rgba(255,255,255,0.82);color:#756b63;'
        'border:1px solid rgba(64,42,31,0.16)">&#10003;</div>'
        '<div style="font-size:11px;font-weight:800;color:#756b63;'
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

    return f"""<div id="wdshell"
     style="font-family:'Segoe UI',system-ui,Arial,sans-serif;
            background:rgba(255,250,244,0.72);
            backdrop-filter:blur(28px) saturate(1.08);
            border:1px solid rgba(255,255,255,0.72);
            border-radius:34px;
            box-shadow:0 34px 90px rgba(52,34,26,0.10),
                       inset 0 1px 0 rgba(255,255,255,0.75);
            overflow:hidden">
{css}
<input type="radio" id="wdt0" name="wdtab"{c[0]} style="display:none">
<input type="radio" id="wdt1" name="wdtab"{c[1]} style="display:none">
<input type="radio" id="wdt2" name="wdtab"{c[2]} style="display:none">
<input type="radio" id="wdt3" name="wdtab"{c[3]} style="display:none">

<!-- ── header ─────────────────────────────────────────────────────── -->
<div id="wdhdr" style="display:grid;grid-template-columns:180px 1fr 160px;
     align-items:start;gap:24px;padding:24px 34px 18px;
     border-bottom:1px solid rgba(63,43,35,0.09)">
  <div>
    <div style="font-family:Georgia,'Times New Roman',serif;font-size:30px;
                letter-spacing:-0.8px;font-weight:700;white-space:nowrap;color:#211917">
      Wine<span style="color:#7a1830">&amp;</span>Dine</div>
    <div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
                color:#42101d;font-weight:800;margin-top:6px">Food &amp; wine pairing</div>
  </div>
  <div style="display:flex;align-items:flex-start;gap:0;
              max-width:680px;margin:0 auto;width:100%">
    {steps_html}
  </div>
  <div style="justify-self:end;align-self:center">
    <label for="wdt0" style="display:block;background:rgba(255,255,255,0.45);
      border:1px solid rgba(63,43,35,0.09);color:#211917;padding:10px 16px;
      border-radius:999px;font-size:13px;font-weight:700;white-space:nowrap;cursor:pointer">
      &#8635; start over
    </label>
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

<div style="padding:10px 34px 14px;font-size:10px;color:#b8aaa0;text-align:right;
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
    """CNN pass — returns screen-1 shell and shows yes/no confirm row."""
    if pil_img is None:
        return "", gr.update(visible=False), gr.update(visible=True), gr.update(), gr.update()

    food_name, conf, top5 = identify_food(pil_img)
    display = food_name.replace("_", " ").title() if "_" in food_name else food_name

    # encode photo as small JPEG for embedding in HTML
    thumb = pil_img.convert("RGB")
    thumb.thumbnail((480, 480))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=82)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    _state.update(food=food_name, conf=conf, top5=top5, img_b64=img_b64)

    s1   = _screen1_html(food_name, conf, top5, img_b64)
    html = _shell_html(s1, "", "", 0, "")
    return (
        gr.update(value=html, visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=f"✓  Yes, it's {display}"),
        gr.update(value="✗  No, correct dish"),
    )


def on_yes():
    """BiLSTM pass — 2 yields: spinner → fully ready (screen 2 + 3 in one shot)."""
    food_name = _state.get("food", "")
    food_key  = food_name.lower().replace(" ", "_")
    conf      = _state["conf"]
    top5      = _state["top5"]
    display   = food_name.replace("_", " ").title() if "_" in food_name else food_name

    s1 = _screen1_html(food_name, conf, top5, _state.get("img_b64", ""))

    # ── Yield 0: show spinner on screen 2 BEFORE BiLSTM runs ────────────────
    yield gr.update(visible=False), _shell_html(s1, _LOADING_SPINNER, "", 1, "")

    # BiLSTM inference (slow)
    cluster_idx, cluster_name, sims, desc, attn_w = bilstm_encode(food_key)
    img_b64 = _state.get("img_b64", "")
    s2 = _screen2_html(display, desc, attn_w, cluster_idx, cluster_name, sims, img_b64)

    # Wine lookup is fast (pre-computed JSON) — build s3 immediately
    recs         = RESULTS_ALL.get(food_key, [])
    safe_cluster = recs[0].get("name", cluster_name) if recs else cluster_name
    feel         = _food_feel(safe_cluster)
    s3           = _screen3_html(display, cluster_name, recs, feel)
    s4           = _screen4_html()

    # ── Yield 1: both screens ready; user reads screen 2, clicks CTA for 3 ──
    yield gr.update(visible=False), _shell_html(s1, s2, s3, 1, s4)


def on_no():
    """Reset to blank state — show upload screen again."""
    _state.update(food="", conf=0.0, top5=[], img_b64="")
    return "", gr.update(visible=False), gr.update(visible=True)


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
  max-width: 1140px !important;
  margin: 0 auto !important;
  padding: 32px 18px !important;
  background: transparent !important;
  min-height: 100vh;
}
footer { display: none !important; }
div.main { padding: 0 !important; background: transparent !important; }
.block, .form {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  padding: 0 !important;
}
.gap { gap: 0 !important; }

/* Upload screen layout */
#wdupload {
  display: flex;
  flex-direction: column !important;
  gap: 14px !important;
}
#wdfood {
  margin-top: 6px !important;
}
#wdfood > .wrap {
  border-radius: 24px !important;
  overflow: visible !important;
  padding-bottom: 10px !important;
}
/* Image upload zone */
#wdfood .upload-container, #wdfood [data-testid="image"] {
  border: 1.5px dashed rgba(122,24,48,0.26) !important;
  border-radius: 24px !important;
  background:
    radial-gradient(circle at 16% 18%, rgba(255,255,255,0.86), transparent 34%),
    linear-gradient(145deg, rgba(255,255,255,0.58), rgba(251,245,238,0.78)) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.72),
    0 10px 28px rgba(64,42,31,0.07) !important;
  min-height: 240px !important;
}
#wdfood .upload-container:hover,
#wdfood [data-testid="image"]:hover {
  border-color: rgba(122,24,48,0.42) !important;
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

/* Upload CTA */
#wduploadcta {
  padding: 12px 0 0 !important;
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
/* Card + confirm row */
#wdcard > .wrap { padding: 0 !important; background: transparent !important; }
#wdconfirm {
  width: min(520px, calc(100% - 68px)) !important;
  margin: 14px 0 0 34px !important;
}
#wdconfirm > .wrap {
  padding: 10px 0 !important;
  gap: 12px !important;
  flex-direction: column !important;
  justify-content: flex-start !important;
  align-items: stretch !important;
}
#wdconfirm > .wrap > * { width: 100% !important; }
"""
_UPLOAD_HEADER_HTML = """
<div style="font-family:'Segoe UI',system-ui,Arial,sans-serif;
            background:rgba(255,250,244,0.72);
            backdrop-filter:blur(28px) saturate(1.08);
            border:1px solid rgba(255,255,255,0.72);
            border-radius:34px;
            box-shadow:0 34px 90px rgba(52,34,26,0.10),inset 0 1px 0 rgba(255,255,255,0.75);
            overflow:hidden">

  <!-- header row: brand | step bar | restart -->
  <div style="display:grid;grid-template-columns:180px 1fr 160px;
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
    <div style="display:grid;grid-template-columns:repeat(9,auto);
                align-items:center;justify-content:center;gap:0;max-width:660px;margin:0 auto">
      <!-- step 1 active -->
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div style="width:40px;height:40px;border-radius:999px;display:grid;
                    place-items:center;background:linear-gradient(135deg,#8d1f3a,#5a1024);
                    color:#fff;font-size:16px;font-weight:900;
                    box-shadow:0 10px 28px rgba(122,24,48,0.24)">1</div>
        <div style="font-size:11px;font-weight:800;color:#42101d;
                    text-decoration:underline;text-underline-offset:4px;white-space:nowrap">Upload</div>
      </div>
      <div style="width:48px;height:1px;background:rgba(64,42,31,0.18);margin-bottom:20px"></div>
      <!-- step 2 -->
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div style="width:40px;height:40px;border-radius:999px;display:grid;
                    place-items:center;background:rgba(255,255,255,0.58);
                    color:#9e9188;font-size:16px;font-weight:900;
                    border:1px solid rgba(200,190,180,0.5)">2</div>
        <div style="font-size:11px;font-weight:800;color:#9e9188;white-space:nowrap">Detect Dish</div>
      </div>
      <div style="width:48px;height:1px;background:rgba(64,42,31,0.18);margin-bottom:20px"></div>
      <!-- step 3 -->
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div style="width:40px;height:40px;border-radius:999px;display:grid;
                    place-items:center;background:rgba(255,255,255,0.58);
                    color:#9e9188;font-size:16px;font-weight:900;
                    border:1px solid rgba(200,190,180,0.5)">3</div>
        <div style="font-size:11px;font-weight:800;color:#9e9188;white-space:nowrap">Fingerprint</div>
      </div>
      <div style="width:48px;height:1px;background:rgba(64,42,31,0.18);margin-bottom:20px"></div>
      <!-- step 4 -->
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div style="width:40px;height:40px;border-radius:999px;display:grid;
                    place-items:center;background:rgba(255,255,255,0.58);
                    color:#9e9188;font-size:16px;font-weight:900;
                    border:1px solid rgba(200,190,180,0.5)">4</div>
        <div style="font-size:11px;font-weight:800;color:#9e9188;white-space:nowrap">Wine Pairings</div>
      </div>
      <div style="width:48px;height:1px;background:rgba(64,42,31,0.18);margin-bottom:20px"></div>
      <!-- step 5 -->
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div style="width:40px;height:40px;border-radius:999px;display:grid;
                    place-items:center;background:rgba(255,255,255,0.58);
                    color:#9e9188;font-size:16px;font-weight:900;
                    border:1px solid rgba(200,190,180,0.5)">5</div>
        <div style="font-size:11px;font-weight:800;color:#9e9188;white-space:nowrap">The Story</div>
      </div>
    </div>
    <div style="justify-self:end;align-self:center">
      <div style="background:rgba(255,255,255,0.45);border:1px solid rgba(63,43,35,0.09);
                  color:#211917;padding:10px 16px;border-radius:999px;
                  font-size:13px;font-weight:700;white-space:nowrap">&#8635; start over</div>
    </div>
  </div>

  <!-- intro copy -->
  <div style="padding:36px 34px 28px">
    <div style="font-size:11px;color:#7a1830;text-transform:uppercase;
                letter-spacing:0.14em;font-weight:800;margin-bottom:10px">STEP 1 &middot; UPLOAD</div>
    <div style="font-family:Georgia,'Times New Roman',serif;font-size:46px;
                font-weight:700;color:#211917;letter-spacing:-2px;line-height:0.95;
                margin-bottom:16px">What are you<br>eating?</div>
    <div style="font-size:14px;color:#7c726b;line-height:1.7;max-width:520px">
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

    # ── Upload screen (screen 0) ──────────────────────────────────────────────
    with gr.Column(visible=True, elem_id="wdupload") as upload_col:
        gr.HTML(_UPLOAD_HEADER_HTML)
        img_input = gr.Image(
            type="pil", label="", show_label=False,
            height=240, elem_id="wdfood",
            sources=["upload", "clipboard"],
        )
        with gr.Row(elem_id="wduploadcta"):
            identify_btn = gr.Button(
                "Analyze dish →", variant="primary",
                elem_id="wdanalyze",
            )

    # ── Result card (screens 1–4) ─────────────────────────────────────────────
    wine_card = gr.HTML(visible=False, elem_id="wdcard")

    with gr.Column(visible=False, elem_id="wdconfirm") as confirm_row:
        yes_btn = gr.Button(
            "✓  Yes, it's my dish",
            variant="primary", elem_id="wdyes",
        )
        no_btn = gr.Button(
            "✗  No, correct dish",
            variant="secondary", elem_id="wdno",
        )

    # ── Event wiring ──────────────────────────────────────────────────────────
    identify_btn.click(
        on_identify,
        inputs=[img_input],
        outputs=[wine_card, confirm_row, upload_col, yes_btn, no_btn],
    )
    yes_btn.click(
        on_yes,
        inputs=None,
        outputs=[confirm_row, wine_card],
    )
    no_btn.click(
        on_no,
        inputs=None,
        outputs=[wine_card, confirm_row, upload_col],
    )

if __name__ == "__main__":
    demo.launch()
