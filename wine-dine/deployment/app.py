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
    """Screen 1 — dish identification result."""
    display  = food_name.replace("_", " ").title()
    conf_pct = int(conf * 100)

    # Photo — real image or gradient placeholder
    if img_b64:
        photo_inner = (
            f'<img src="data:image/jpeg;base64,{img_b64}"'
            f' style="width:100%;height:100%;object-fit:cover;display:block">'
        )
    else:
        photo_inner = (
            '<div style="width:100%;height:100%;display:flex;align-items:center;'
            'justify-content:center;font-size:56px">🍽️</div>'
        )

    # Top-5 bar rows (skipping the top prediction — shown as hero)
    others_rows = ""
    for fn, fp in top5[1:]:
        bar_w = int(fp * 180)
        others_rows += (
            f'<div style="display:grid;grid-template-columns:96px 1fr 38px;'
            f'gap:10px;align-items:center;font-size:12px;margin-bottom:7px">'
            f'<span style="color:#756b63;overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap">{fn}</span>'
            f'<div style="height:7px;background:#efe8df;border-radius:999px;overflow:hidden">'
            f'<div style="background:#c69b55;width:{bar_w}px;height:100%;border-radius:999px">'
            f'</div></div>'
            f'<span style="color:#756b63;font-weight:700">{fp*100:.1f}%</span>'
            f'</div>'
        )

    return f"""
<div style="display:grid;grid-template-columns:1.05fr 0.95fr;gap:32px;align-items:center">

  <!-- photo -->
  <div style="border-radius:18px;overflow:hidden;min-height:320px;
              background:linear-gradient(rgba(40,16,18,0.06),rgba(40,16,18,0.18)),
                         #ede3d8">
    {photo_inner}
  </div>

  <!-- right panel -->
  <div style="display:flex;flex-direction:column;align-items:center;text-align:center">
    <div style="font-size:12px;color:#756b63;text-transform:uppercase;
                letter-spacing:0.12em;font-weight:800;margin-bottom:6px">we think this is</div>

    <div style="font-family:Georgia,'Times New Roman',serif;
                font-size:62px;line-height:0.9;letter-spacing:-2px;
                color:#4a1020;margin-bottom:18px">{display}</div>

    <!-- confidence bar -->
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;
                width:100%;max-width:300px;justify-content:center">
      <div style="flex:1;height:10px;background:#efe8df;border-radius:999px;overflow:hidden">
        <div style="background:#7a1f32;width:{conf_pct}%;height:100%;border-radius:999px"></div>
      </div>
      <strong style="color:#7a1f32;font-family:Georgia,serif;font-size:20px">{conf_pct}%</strong>
    </div>

    <!-- other possibilities -->
    <div style="width:100%;max-width:340px;padding:14px 16px;
                border:1px solid rgba(64,42,31,0.14);border-radius:18px;
                background:rgba(251,245,238,0.8);text-align:left;margin-bottom:22px">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.12em;
                  font-weight:800;color:#b8aaa0;margin-bottom:10px">other possibilities</div>
      {others_rows}
    </div>

    <!-- confirm / correct labels — wired by confirm_row below the shell -->
    <div style="font-size:12px;color:#b8aaa0;font-style:italic">
      use the buttons below to confirm or correct
    </div>
  </div>
</div>"""


def _screen2_html(display: str, desc: str, attn_w,
                  cluster_idx: int, cluster_name: str, sims,
                  img_b64: str = "") -> str:
    """Screen 2 — taste fingerprint (matches mockup Screen 2)."""

    # Attention-highlighted words — wine-red palette (3 heat levels)
    words     = desc.split()[:MAX_SEQ_LEN]
    attn_arr  = attn_w[:len(words)]
    a_min, a_max = attn_arr.min(), attn_arr.max()
    attn_norm = (attn_arr - a_min) / (a_max - a_min + 1e-8)
    word_html = ""
    for w_txt, a in zip(words, attn_norm):
        if a >= 0.75:
            style = "background:#7a1f32;color:#fff;font-weight:800"   # hot
        elif a >= 0.40:
            style = "background:rgba(198,155,85,0.32);font-weight:600"  # warm gold
        elif a >= 0.15:
            style = "background:rgba(122,31,50,0.08)"                  # faint blush
        else:
            style = ""
        word_html += (
            f'<span style="border-radius:3px;padding:1px 4px;margin:1px;'
            f'font-size:13px;{style}">{w_txt}</span> '
        )

    # Cluster bars — top-5 by cosine sim
    sorted_k  = np.argsort(sims)[::-1][:5] if len(sims) > 0 else []
    cluster_rows = ""
    for k in sorted_k:
        sim_val = float(sims[k])
        bar_w   = int(sim_val * 200)
        is_top  = int(k) == cluster_idx
        name_fw = "800" if is_top else "400"
        name_c  = "#4a1020" if is_top else "#756b63"
        bar_c   = "#7a1f32" if is_top else "#ddd5cc"
        cluster_rows += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
          <span style="font-size:12px;color:{name_c};font-weight:{name_fw};
                       width:210px;flex-shrink:0;overflow:hidden;
                       white-space:nowrap;text-overflow:ellipsis">
            {CLUSTER_NAMES.get(int(k), str(k))}</span>
          <div style="flex:1;height:4px;background:#edecea;border-radius:2px;overflow:hidden">
            <div style="background:{bar_c};width:{bar_w}px;height:4px;border-radius:2px"></div>
          </div>
          <span style="font-size:11px;color:#9a9895;width:38px;text-align:right">{sim_val:.3f}</span>
        </div>"""

    info_text = (
        "TasteBiLSTM + Bahdanau attention &middot; 10 sensory axes: acidity &middot; "
        "tannin &middot; red fruit &middot; dark fruit &middot; earthy &middot; sweet &middot; "
        "body &middot; oaky &middot; floral &middot; mineral &middot; 512-d L2-normalised "
        "taste vector &middot; BisectingKMeans K=9 &middot; TF-IDF cluster naming"
    )

    # CSS for the info toggle (checkbox trick — no JS needed)
    css = (
        "<style>"
        "#wdinf2panel{display:none}"
        "#wdinf2:checked~#wdinf2panel{display:block}"
        "</style>"
    )

    # Compact food-photo context strip
    if img_b64:
        photo_strip = (
            f'<div style="display:flex;align-items:center;gap:14px;'
            f'padding:12px 16px;background:rgba(255,250,243,0.9);'
            f'border-radius:14px;border:1px solid rgba(64,42,31,0.12);'
            f'margin-bottom:18px">'
            f'<img src="data:image/jpeg;base64,{img_b64}"'
            f' style="width:56px;height:56px;border-radius:10px;'
            f'object-fit:cover;flex-shrink:0">'
            f'<div>'
            f'<div style="font-size:10px;color:#b8aaa0;text-transform:uppercase;'
            f'letter-spacing:0.12em;font-weight:800">analyzing</div>'
            f'<div style="font-family:Georgia,serif;font-size:22px;'
            f'color:#4a1020;letter-spacing:-0.5px;line-height:1.1">{display}</div>'
            f'</div>'
            f'</div>'
        )
    else:
        photo_strip = ""

    return f"""{css}
{photo_strip}<input type="checkbox" id="wdinf2" style="display:none">
<div style="display:flex;justify-content:space-between;align-items:flex-start;
            margin-bottom:0.75rem">
  <div style="font-size:13px;color:#5a5855;line-height:1.7">
    Your dish's fingerprint was
    <strong style="color:#1a1917">written by Claude Sonnet</strong>
    &mdash; then run through our
    <strong style="color:#1a1917">TasteBiLSTM</strong>
    to understand how it truly tastes.
  </div>
  <label for="wdinf2"
         style="margin-left:10px;flex-shrink:0;width:18px;height:18px;border-radius:50%;
                display:flex;align-items:center;justify-content:center;
                border:1px solid rgba(122,31,50,0.35);color:#7a1f32;font-size:10px;cursor:pointer;
                font-style:italic;font-weight:600;line-height:1">i</label>
</div>
<div id="wdinf2panel"
     style="background:#f0ede8;border-radius:6px;padding:8px 12px;
            font-size:10px;color:#5a5855;line-height:1.7;margin-bottom:0.75rem">
  {info_text}
</div>
<div style="display:flex;gap:0;margin-bottom:1rem">
  <div style="flex:1;padding:9px 12px;border:0.5px solid rgba(26,25,23,0.10);
              background:#f6f5f1;border-radius:8px 0 0 8px;border-right:none">
    <div style="font-size:9px;color:#9a9895;text-transform:uppercase;
                letter-spacing:0.05em;margin-bottom:3px">step 1</div>
    <div style="font-size:12px;font-weight:600;color:#1a1917">flavor description</div>
    <div style="font-size:10px;color:#9a9895;margin-top:2px">written by Claude Sonnet</div>
  </div>
  <div style="display:flex;align-items:center;padding:0 8px;
              background:#f6f5f1;border-top:0.5px solid rgba(26,25,23,0.10);
              border-bottom:0.5px solid rgba(26,25,23,0.10);
              color:#9a9895;font-size:14px">&rarr;</div>
  <div style="flex:1;padding:9px 12px;border:0.5px solid rgba(26,25,23,0.10);
              background:#f6f5f1;border-radius:0 8px 8px 0">
    <div style="font-size:9px;color:#9a9895;text-transform:uppercase;
                letter-spacing:0.05em;margin-bottom:3px">step 2</div>
    <div style="font-size:12px;font-weight:600;color:#1a1917">TasteBiLSTM</div>
    <div style="font-size:10px;color:#9a9895;margin-top:2px">encodes it into a taste vector</div>
  </div>
</div>
<div style="background:#f6f5f1;border-radius:8px;padding:12px 14px;
            font-size:13px;line-height:2.2;border:0.5px solid rgba(26,25,23,0.10);
            margin-bottom:5px">
  {word_html}
</div>
<div style="font-size:10px;color:#9a9895;margin-bottom:1.25rem">
  word warmth = attention weight &middot; more golden = more attended to by the model
</div>
<div style="font-size:10px;color:#9a9895;text-transform:uppercase;
            letter-spacing:0.06em;margin-bottom:8px">
  closeness to each flavor world
</div>
{cluster_rows}
<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
            background:rgba(122,31,50,0.06);border-radius:8px;border-left:3px solid #7a1f32;
            margin-top:4px;margin-bottom:1.5rem">
  <span style="font-size:11px;color:#756b63">taste fingerprint</span>
  <span style="font-size:14px;font-weight:800;color:#4a1020">{cluster_name}</span>
</div>
<div style="text-align:center">
  <label for="wdt2"
         style="display:inline-block;padding:11px 32px;border-radius:999px;
                background:#7a1f32;color:#fff;font-size:14px;font-weight:900;
                cursor:pointer;letter-spacing:-0.2px;border:none;
                box-shadow:0 10px 22px rgba(122,31,50,0.22)">
    find wine pairings &rarr;
  </label>
</div>"""


_TIER_INFO = {
    "SAFE BET":   "nearest cluster centroid by cosine similarity to dominant flavor description",
    "HIDDEN GEM": "nearest centroid to the \u2018surprising pairing\u2019 flavor description",
    "BOLD MOVE":  "score = distance \u00d7 (1 + secondary affinity) \u00b7 maximises contrast while keeping the wine drinkable",
}
_WINE_INFO = "highest cosine similarity between food taste vector and wine taste vectors within this cluster\u2019s 10-wine pool"


def _info_btn(panel_id: str, color: str) -> str:
    return (
        f'<button onclick="var p=document.getElementById(\'{panel_id}\'),'
        f'open=p.style.display===\'block\';'
        f'p.style.display=open?\'none\':\'block\'" '
        f'style="width:16px;height:16px;border-radius:50%;border:1px solid {color};'
        f'background:transparent;color:{color};font-size:9px;cursor:pointer;'
        f'display:inline-flex;align-items:center;justify-content:center;'
        f'flex-shrink:0;line-height:1;padding:0">i</button>'
    )


def _screen3_html(display: str, cluster_name: str, recs: list, feel: str) -> str:
    """Screen 3 — 3-column wine card grid with info buttons (matches mockup Screen 3)."""

    intro = (
        f'<div style="font-size:13px;color:#5a5855;margin-bottom:1rem;line-height:1.6">'
        f'Your {display.lower()}\u2019s fingerprint is '
        f'<strong style="color:#5558A8">{cluster_name}</strong>. '
        f'We found wines that match it \u2014 each from a different angle.</div>'
    )

    cards_html = ""
    for rec in recs[:3]:
        tier     = rec.get("tier", "")
        name     = rec.get("name", "")
        wine     = rec.get("wine", "\u2014")
        rating   = rec.get("rating", "\u2014")
        snippet  = _clip(rec.get("snippet", ""), 120)
        kws      = rec.get("keywords", [])
        conf     = float(rec.get("confidence", 0.0))

        color    = _TIER_COLOR.get(tier, "#555")
        strip_bg = _TIER_STRIP_BG.get(tier, "#f5f5f5")
        icon     = _TIER_ICON.get(tier, "")
        conf_lbl = _TIER_CONF_LABEL.get(tier, "match")
        tag_bg   = _TIER_TAG_BG.get(tier, "#eee")
        tier_lbl = tier.lower()
        adj      = _cluster_adj(name)

        # unique IDs for info panels (avoid collisions if card re-rendered)
        uid       = tier.lower().replace(" ", "-")
        tier_pid  = f"wd-info-{uid}"
        wine_pid  = f"wd-wine-{uid}"

        conf_pct  = int(conf * 100)
        conf_bar_w = int(conf * 110)

        reasoning = (
            f"Your {display.lower()} is {feel} \u2014 "
            + ("this wine matches that energy exactly." if tier == "SAFE BET"
               else "this wine finds an angle most pairings overlook." if tier == "HIDDEN GEM"
               else "this wine goes against it entirely. Sometimes contrast is the pairing.")
        )

        tags_html = ""
        for i, kw in enumerate(kws[:3]):
            if i > 0:
                tags_html += '<span style="font-size:10px;color:#aaa">\u00b7</span>'
            tags_html += (
                f'<span style="font-size:10px;font-weight:500;padding:2px 7px;'
                f'border-radius:20px;background:{tag_bg};color:{color}">{kw}</span>'
            )

        cards_html += f"""
        <div style="border-radius:12px;overflow:hidden;background:#fff;
                    border:0.5px solid rgba(0,0,0,0.08);
                    box-shadow:0 1px 4px rgba(0,0,0,0.05)">
          <div style="background:{strip_bg};padding:9px 11px;
                      display:flex;justify-content:space-between;align-items:flex-start">
            <div style="font-size:10px;font-weight:600;letter-spacing:0.05em;
                        color:{color}">{icon} {tier_lbl}</div>
            <div style="display:flex;align-items:flex-start;gap:5px">
              <div>
                <div style="font-size:19px;font-weight:600;color:{color};line-height:1">{conf_pct}%</div>
                <div style="font-size:9px;color:#aaa">{conf_lbl}</div>
              </div>
              {_info_btn(tier_pid, color)}
            </div>
          </div>
          <div id="{tier_pid}" style="display:none;padding:7px 11px;font-size:10.5px;
                                      line-height:1.55;background:{strip_bg};color:{color};
                                      font-family:monospace">{_TIER_INFO.get(tier,"")}</div>
          <div style="padding:10px 11px;display:flex;flex-direction:column;gap:8px">
            <div style="font-size:11px;color:#5a5855;font-style:italic;line-height:1.55">
              {reasoning}
            </div>
            <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
              <span style="font-size:10px;color:#9a9895">plays on</span>
              {tags_html}
            </div>
            <div style="height:0.5px;background:rgba(0,0,0,0.07)"></div>
            <div style="display:flex;align-items:flex-start;gap:5px">
              <div style="font-size:12px;font-weight:600;color:#1a1917;
                          line-height:1.35;flex:1">{wine}</div>
              {_info_btn(wine_pid, "#9a9895")}
            </div>
            <div id="{wine_pid}" style="display:none;font-size:10px;color:#9a9895;
                                        font-family:monospace;line-height:1.5">{_WINE_INFO}</div>
            <div style="display:flex;align-items:center;gap:4px">
              <span style="color:#E6A817;font-size:11px">\u2605</span>
              <span style="font-size:11px;font-weight:500;color:#1a1917">{rating}</span>
              <span style="font-size:10px;color:#9a9895">/ 100</span>
            </div>
            <div style="font-size:10.5px;color:#5a5855;font-style:italic;line-height:1.5;
                        border-left:2px solid rgba(0,0,0,0.08);padding-left:7px">
              \u201c{snippet}\u201d
            </div>
          </div>
        </div>"""

    return f"""
    <div id="wd-screen3">
      {intro}
      <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px">
        {cards_html}
      </div>
    </div>"""


def _shell_html(s1: str, s2: str, s3: str, cur: int) -> str:
    """
    Full 3-screen shell using CSS radio-tab navigation — zero JavaScript.
    cur = 0/1/2 — which tab is checked on server render.
    """
    unlocked = int(bool(s1)) + int(bool(s2)) + int(bool(s3)) - 1
    c = [" checked" if i == cur else "" for i in range(3)]

    # ── Step bar (numbered circles + connector lines) ─────────────────────────
    step_labels = [
        ("1", "Identify dish"),
        ("2", "Taste fingerprint"),
        ("3", "Wine pairings"),
    ]
    steps_html = ""
    for i, (num, lbl) in enumerate(step_labels):
        if i <= unlocked:
            circle = (
                f'<label for="wdt{i}" class="wddt wddt{i}"'
                f' style="width:40px;height:40px;border-radius:50%;display:grid;'
                f'place-items:center;font-size:15px;font-weight:900;cursor:pointer;'
                f'box-sizing:border-box;position:relative;z-index:2;'
                f'border:1px solid rgba(64,42,31,0.16);background:rgba(255,255,255,0.82);'
                f'color:#756b63;box-shadow:0 4px 14px rgba(66,37,20,0.08);'
                f'transition:all 0.15s">{num}</label>'
            )
        else:
            circle = (
                f'<div style="width:40px;height:40px;border-radius:50%;display:grid;'
                f'place-items:center;font-size:15px;font-weight:900;'
                f'box-sizing:border-box;position:relative;z-index:2;'
                f'border:1px solid rgba(64,42,31,0.08);background:rgba(255,255,255,0.45);'
                f'color:#ccc">{num}</div>'
            )
        # label text (shown under circle)
        lbl_c = "#4a1020" if i == cur else ("#756b63" if i < cur else "#b8aaa0")
        lbl_fw = "900" if i == cur else "600"
        lbl_dec = "underline;text-underline-offset:3px" if i == cur else "none"

        steps_html += (
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:8px">'
            f'{circle}'
            f'<div style="font-size:11px;font-weight:{lbl_fw};color:{lbl_c};'
            f'text-align:center;text-decoration:{lbl_dec};white-space:nowrap">{lbl}</div>'
            f'</div>'
        )
        if i < 2:
            steps_html += (
                '<div style="flex:1;height:1px;background:rgba(64,42,31,0.16);'
                'margin-bottom:22px;align-self:center;position:relative;z-index:0"></div>'
            )

    # ── Nav buttons ───────────────────────────────────────────────────────────
    _btn_back = ("padding:10px 20px;border-radius:999px;font-size:13px;display:none;"
                 "border:1px solid rgba(64,42,31,0.18);background:rgba(255,255,255,0.45);';"
                 "color:#756b63;cursor:pointer;font-weight:700")
    _btn_next = ("padding:10px 20px;border-radius:999px;font-size:13px;display:none;"
                 "border:none;background:#7a1f32;color:#fff;cursor:pointer;font-weight:900;"
                 "box-shadow:0 10px 24px rgba(122,31,50,0.23)")
    _lock      = "pointer-events:none;opacity:0.28"
    _ctr       = ("font-size:11px;color:#756b63;display:none;text-transform:uppercase;"
                  "letter-spacing:0.12em;font-weight:800")

    # ── CSS (radio-tab logic, active/done dot styles) ─────────────────────────
    css = (
        "<style>"
        "#wds0,#wds1,#wds2{display:none}"
        "#wdt0:checked~#wds0,#wdt1:checked~#wds1,#wdt2:checked~#wds2{display:block}"
        # Active step circle — wine red fill + glow
        "#wdt0:checked~#wdprog .wddt0,"
        "#wdt1:checked~#wdprog .wddt1,"
        "#wdt2:checked~#wdprog .wddt2"
        "{background:#7a1f32!important;color:#fff!important;"
        "border-color:#7a1f32!important;"
        "box-shadow:0 10px 24px rgba(122,31,50,0.24)!important}"
        # Done step circle — muted white
        "#wdt1:checked~#wdprog .wddt0,"
        "#wdt2:checked~#wdprog .wddt0,"
        "#wdt2:checked~#wdprog .wddt1"
        "{background:rgba(255,255,255,0.82)!important;"
        "color:#756b63!important;border-color:rgba(64,42,31,0.16)!important;"
        "box-shadow:none!important}"
        # Back labels
        "#wdt1:checked~#wdnav #wdb1,"
        "#wdt2:checked~#wdnav #wdb2{display:inline-block}"
        # Next labels
        "#wdt0:checked~#wdnav #wdn0,"
        "#wdt1:checked~#wdnav #wdn1,"
        "#wdt2:checked~#wdnav #wdn2{display:inline-block}"
        # Counters
        "#wdt0:checked~#wdnav #wdc0,"
        "#wdt1:checked~#wdnav #wdc1,"
        "#wdt2:checked~#wdnav #wdc2{display:inline}"
        # Progress pills in nav — active = wine red + wider
        "#wdt0:checked~* .wddt0,"
        "#wdt1:checked~* .wddt1,"
        "#wdt2:checked~* .wddt2"
        "{background:#7a1f32!important;width:28px!important}"
        "</style>"
    )

    n0_lock = _lock if unlocked < 1 else ""
    n1_lock = _lock if unlocked < 2 else ""

    return f"""<div id="wdshell"
     style="font-family:'Segoe UI',system-ui,Arial,sans-serif;
            background:radial-gradient(circle at 18% 8%,rgba(189,104,70,0.13),transparent 34%),
                       radial-gradient(circle at 88% 18%,rgba(122,31,50,0.09),transparent 34%),
                       linear-gradient(135deg,#f8f3eb 0%,#eee3d6 100%);
            border-radius:30px;padding:0;max-width:760px;margin:0 auto;
            box-shadow:0 24px 70px rgba(66,37,20,0.14);
            border:1px solid rgba(255,255,255,0.62);overflow:hidden">
{css}
<input type="radio" id="wdt0" name="wdtab"{c[0]} style="display:none">
<input type="radio" id="wdt1" name="wdtab"{c[1]} style="display:none">
<input type="radio" id="wdt2" name="wdtab"{c[2]} style="display:none">

<!-- ── header ───────────────────────────────────────────────────── -->
<div style="display:grid;grid-template-columns:140px 1fr 120px;
            align-items:center;gap:16px;
            padding:22px 32px 18px;
            border-bottom:1px solid rgba(64,42,31,0.10);
            background:rgba(255,250,243,0.55)">
  <div>
    <div style="font-family:Georgia,'Times New Roman',serif;font-size:26px;
                font-weight:700;color:#231b17;letter-spacing:-0.8px;line-height:1">
      Wine<span style="color:#7a1f32">&amp;</span>Dine</div>
    <div style="font-size:10px;letter-spacing:0.12em;text-transform:uppercase;
                color:#7a1f32;font-weight:800;margin-top:4px">AI pairing</div>
  </div>

  <!-- step bar -->
  <div id="wdprog"
       style="display:flex;align-items:flex-start;justify-content:center;gap:0;
              max-width:480px;margin:0 auto;width:100%">
    {steps_html}
  </div>

  <!-- restart -->
  <label for="wdt0"
         style="justify-self:end;padding:9px 14px;border-radius:999px;font-size:12px;
                border:1px solid rgba(64,42,31,0.18);background:rgba(255,255,255,0.45);
                color:#231b17;cursor:pointer;font-weight:700;white-space:nowrap">
    &#8635; start over
  </label>
</div>

<!-- ── screens ──────────────────────────────────────────────────── -->
<div style="padding:28px 32px 24px;background:rgba(255,250,243,0.82)">
  <div id="wds0">{s1 or ""}</div>
  <div id="wds1">{s2 or _LOADING_SPINNER}</div>
  <div id="wds2">{s3 or ""}</div>
</div>

<!-- ── nav bar ──────────────────────────────────────────────────── -->
<div id="wdnav"
     style="display:flex;justify-content:space-between;align-items:center;
            padding:16px 32px;
            border-top:1px solid rgba(64,42,31,0.10);
            background:rgba(255,250,243,0.72)">
  <div>
    <label for="wdt0" id="wdb1" style="{_btn_back}">&larr; Back</label>
    <label for="wdt1" id="wdb2" style="{_btn_back}">&larr; Back</label>
  </div>
  <div style="text-align:center">
    <span id="wdc0" style="{_ctr}">Step 1 of 3</span>
    <span id="wdc1" style="{_ctr}">Step 2 of 3</span>
    <span id="wdc2" style="{_ctr}">Step 3 of 3</span>
    <div style="display:flex;gap:6px;justify-content:center;margin-top:7px">
      <span class="wddt wddt0"
            style="display:inline-block;width:24px;height:5px;border-radius:999px;
                   background:rgba(122,31,50,0.18)"></span>
      <span class="wddt wddt1"
            style="display:inline-block;width:24px;height:5px;border-radius:999px;
                   background:rgba(122,31,50,0.18)"></span>
      <span class="wddt wddt2"
            style="display:inline-block;width:24px;height:5px;border-radius:999px;
                   background:rgba(122,31,50,0.18)"></span>
    </div>
  </div>
  <div>
    <label for="wdt1" id="wdn0" style="{_btn_next};{n0_lock}">Next &rarr;</label>
    <label for="wdt2" id="wdn1" style="{_btn_next};{n1_lock}">Next &rarr;</label>
    <span   id="wdn2" style="{_btn_next}">Done &#10003;</span>
  </div>
</div>

<div style="padding:10px 32px 14px;background:rgba(255,250,243,0.72);
            font-size:10px;color:#b8aaa0;text-align:right;
            border-top:1px solid rgba(64,42,31,0.06)">
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

    # ── Yield 2: all done, advance to screen3 ───────────────────────────────
    yield _shell_html(s1, s2, s3, 2)



# ── App state ─────────────────────────────────────────────────────────────────
_state: dict = {"food": "", "conf": 0.0, "top5": [], "img_b64": ""}

# ── Event handlers ─────────────────────────────────────────────────────────────
def on_identify(pil_img):
    """CNN pass — returns screen-1 shell and shows yes/no confirm row."""
    if pil_img is None:
        return "", gr.update(visible=False), gr.update(visible=True)

    food_name, conf, top5 = identify_food(pil_img)

    # encode photo as small JPEG for embedding in HTML
    thumb = pil_img.convert("RGB")
    thumb.thumbnail((480, 480))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=82)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    _state.update(food=food_name, conf=conf, top5=top5, img_b64=img_b64)

    s1   = _screen1_html(food_name, conf, top5, img_b64)
    html = _shell_html(s1, "", "", 0)
    # hide upload screen, show result card + confirm buttons
    return gr.update(value=html, visible=True), gr.update(visible=True), gr.update(visible=False)


def on_yes():
    """BiLSTM pass — 2 yields: spinner → fully ready (screen 2 + 3 in one shot)."""
    food_name = _state.get("food", "")
    food_key  = food_name.lower().replace(" ", "_")
    conf      = _state["conf"]
    top5      = _state["top5"]
    display   = food_name.replace("_", " ").title() if "_" in food_name else food_name

    s1 = _screen1_html(food_name, conf, top5, _state.get("img_b64", ""))

    # ── Yield 0: show spinner on screen 2 BEFORE BiLSTM runs ────────────────
    yield gr.update(visible=False), _shell_html(s1, _LOADING_SPINNER, "", 1)

    # BiLSTM inference (slow)
    cluster_idx, cluster_name, sims, desc, attn_w = bilstm_encode(food_key)
    img_b64 = _state.get("img_b64", "")
    s2 = _screen2_html(display, desc, attn_w, cluster_idx, cluster_name, sims, img_b64)

    # Wine lookup is fast (pre-computed JSON) — build s3 immediately
    recs         = RESULTS_ALL.get(food_key, [])
    safe_cluster = recs[0].get("name", cluster_name) if recs else cluster_name
    feel         = _food_feel(safe_cluster)
    s3           = _screen3_html(display, cluster_name, recs, feel)

    # ── Yield 1: both screens ready; user reads screen 2, clicks CTA for 3 ──
    yield gr.update(visible=False), _shell_html(s1, s2, s3, 1)


def on_no():
    """Reset to blank state — show upload screen again."""
    _state.update(food="", conf=0.0, top5=[], img_b64="")
    return "", gr.update(visible=False), gr.update(visible=True)


# ── Full-screen app CSS ────────────────────────────────────────────────────────
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
/* Image upload zone */
#wdfood .upload-container, #wdfood [data-testid="image"] {
  border: 1.5px dashed rgba(122,24,48,0.22) !important;
  border-radius: 20px !important;
  background: linear-gradient(145deg,rgba(255,255,255,0.56),rgba(251,245,238,0.72)) !important;
  min-height: 220px !important;
}
/* Analyze button */
#wdanalyze button {
  background: linear-gradient(135deg,#8d1f3a 0%,#5a1024 100%) !important;
  color: #fff !important; border: none !important; border-radius: 999px !important;
  font-size: 15px !important; font-weight: 900 !important;
  padding: 14px 32px !important;
  box-shadow: 0 14px 30px rgba(122,24,48,0.22) !important;
  letter-spacing: -0.2px !important;
}
#wdanalyze button:hover { opacity: 0.88 !important; }
/* Confirm buttons */
#wdyes button {
  background: linear-gradient(135deg,#8d1f3a 0%,#5a1024 100%) !important;
  color: #fff !important; border: none !important; border-radius: 999px !important;
  font-weight: 900 !important; padding: 13px 28px !important;
  box-shadow: 0 10px 24px rgba(122,24,48,0.22) !important;
}
#wdno button {
  background: rgba(255,255,255,0.62) !important; color: #42101d !important;
  border: 1px solid rgba(122,24,48,0.30) !important; border-radius: 999px !important;
  font-weight: 800 !important; padding: 13px 24px !important;
}
/* Card + confirm row inner padding */
#wdcard > .wrap { padding: 0 !important; background: transparent !important; }
#wdconfirm > .wrap { padding: 16px 0 0 !important; gap: 12px !important; }
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
                  color:#42101d;font-weight:800;margin-top:6px">AI food &amp; wine pairing</div>
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
    with gr.Column(visible=True) as upload_col:
        gr.HTML(_UPLOAD_HEADER_HTML)
        img_input = gr.Image(
            type="pil", label="", show_label=False,
            height=240, elem_id="wdfood",
            sources=["upload", "clipboard"],
        )
        with gr.Row():
            identify_btn = gr.Button(
                "Analyze dish →", variant="primary",
                elem_id="wdanalyze", scale=0, min_width=180,
            )

    # ── Result card (screens 1–4) ─────────────────────────────────────────────
    wine_card = gr.HTML(visible=False, elem_id="wdcard")

    with gr.Row(visible=False, elem_id="wdconfirm") as confirm_row:
        yes_btn = gr.Button(
            "✓  Yes, that's my dish — show pairings!",
            variant="primary", elem_id="wdyes",
        )
        no_btn = gr.Button(
            "↩  Not quite, try again",
            variant="secondary", elem_id="wdno",
        )

    # ── Event wiring ──────────────────────────────────────────────────────────
    identify_btn.click(
        on_identify,
        inputs=[img_input],
        outputs=[wine_card, confirm_row, upload_col],
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
