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
_TIER_COLOR = {
    "SAFE BET":   "#2E7D52",
    "HIDDEN GEM": "#3C3489",
    "BOLD MOVE":  "#993C1D",
}

_TIER_STRIP_BG = {
    "SAFE BET":   "#EAF3DE",
    "HIDDEN GEM": "#EEEDFE",
    "BOLD MOVE":  "#FAECE7",
}

_TIER_ICON = {
    "SAFE BET":   "✅",
    "HIDDEN GEM": "💎",
    "BOLD MOVE":  "🔥",
}

_TIER_CONF_LABEL = {
    "SAFE BET":   "match",
    "HIDDEN GEM": "match",
    "BOLD MOVE":  "contrast",
}

_TIER_TAG_BG = {
    "SAFE BET":   "#EAF3DE",
    "HIDDEN GEM": "#EEEDFE",
    "BOLD MOVE":  "#FAECE7",
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


def _conf_bar_html(conf: float, color: str, label: str = "match") -> str:
    pct = int(conf * 100)
    bar_w = int(conf * 110)
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        f'<div style="background:#e0ddd8;border-radius:3px;width:110px;height:5px;overflow:hidden">'
        f'<div style="background:{color};width:{bar_w}px;height:5px;border-radius:3px"></div>'
        f'</div>'
        f'<span style="font-size:19px;font-weight:600;color:{color};line-height:1">{pct}%</span>'
        f'<span style="font-size:9px;color:#aaa">{label}</span>'
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


def _screen1_html(food_name: str, conf: float, top5: list) -> str:
    """Screen 1 — dish identification result (matches mockup Screen 1)."""
    display = food_name.replace("_", " ").title()
    conf_pct = int(conf * 100)
    conf_bar_w = int(conf * 110)

    # Top-5 prediction bars
    pred_rows = ""
    for fn, fp in top5:
        bar_w   = int(fp * 200)
        is_top  = fn == food_name
        name_fw = "600" if is_top else "400"
        name_c  = "#1a1917" if is_top else "#5a5855"
        bar_c   = "#B85C38" if is_top else "#d4cfc9"
        pred_rows += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px">
          <span style="font-size:12px;color:{name_c};font-weight:{name_fw};
                       width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{fn}</span>
          <div style="flex:1;height:4px;background:#edecea;border-radius:2px;overflow:hidden">
            <div style="background:{bar_c};width:{bar_w}px;height:4px;border-radius:2px"></div>
          </div>
          <span style="font-size:11px;color:#9a9895;width:34px;text-align:right">{fp*100:.0f}%</span>
        </div>"""

    return f"""
    <div id="wd-screen1">
      <div style="font-size:11px;color:#9a9895;margin-bottom:1rem;line-height:1.7">
        We think we know what's on your plate. Does this look right?
      </div>
      <div style="display:flex;gap:16px;align-items:center;margin-bottom:1.5rem">
        <div style="width:86px;height:86px;border-radius:12px;background:#edecea;
                    border:0.5px solid rgba(26,25,23,0.10);display:flex;
                    align-items:center;justify-content:center;font-size:40px;flex-shrink:0">
          🍽️
        </div>
        <div>
          <div style="font-size:11px;color:#9a9895;margin-bottom:4px">detected dish</div>
          <div style="font-size:26px;font-weight:600;color:#1a1917;letter-spacing:-0.4px">{display}</div>
          <div style="display:flex;align-items:center;gap:8px;margin-top:5px">
            <div style="width:110px;height:5px;background:#edecea;border-radius:3px;overflow:hidden">
              <div style="background:#6BAA75;width:{conf_bar_w}px;height:5px;border-radius:3px"></div>
            </div>
            <span style="font-size:12px;font-weight:500;color:#6BAA75">{conf_pct}% confident</span>
          </div>
        </div>
      </div>
      <div style="font-size:10px;color:#9a9895;text-transform:uppercase;
                  letter-spacing:0.06em;margin-bottom:8px">top-5 predictions</div>
      {pred_rows}
    </div>"""


def _screen2_html(display: str, desc: str, attn_w,
                  cluster_idx: int, cluster_name: str, sims) -> str:
    """Screen 2 — taste fingerprint (matches mockup Screen 2)."""

    # Attention-highlighted words — three opacity levels like mockup
    words     = desc.split()[:MAX_SEQ_LEN]
    attn_arr  = attn_w[:len(words)]
    a_min, a_max = attn_arr.min(), attn_arr.max()
    attn_norm = (attn_arr - a_min) / (a_max - a_min + 1e-8)
    word_html = ""
    for w_txt, a in zip(words, attn_norm):
        if a >= 0.75:
            style = "background:#7A82D0;color:#fff;font-weight:600"
        elif a >= 0.40:
            style = "background:rgba(122,130,208,0.32)"
        elif a >= 0.15:
            style = "background:rgba(122,130,208,0.15)"
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
        name_fw = "600" if is_top else "400"
        name_c  = "#1a1917" if is_top else "#5a5855"
        bar_c   = "#7A82D0" if is_top else "#d4cfc8"
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

    return f"""
    <div id="wd-screen2">
      <div style="font-size:13px;color:#5a5855;line-height:1.7;margin-bottom:1rem">
        Your dish's flavor fingerprint was encoded by our
        <strong style="color:#1a1917">TasteBiLSTM</strong>
        to understand how it truly tastes.
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
                    color:#9a9895;font-size:14px">→</div>
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
        highlighted words = what the model pays most attention to
      </div>
      <div style="font-size:10px;color:#9a9895;text-transform:uppercase;
                  letter-spacing:0.06em;margin-bottom:8px">
        how close is your {display.lower()} to each flavor world?
      </div>
      {cluster_rows}
      <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
                  background:#f6f5f1;border-radius:8px;border-left:3px solid #7A82D0;
                  margin-top:4px">
        <span style="font-size:11px;color:#9a9895">taste fingerprint</span>
        <span style="font-size:14px;font-weight:600;color:#5558A8">{cluster_name}</span>
      </div>
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
    Full 3-screen shell with progress dots and nav buttons.
    cur = 0/1/2 — which screen is currently active.
    s2/s3 may be empty strings (not yet computed).
    """
    unlocked = int(bool(s1)) + int(bool(s2)) + int(bool(s3)) - 1  # highest unlocked index

    screens_html = ""
    for idx, content in enumerate([s1, s2, s3]):
        display = "block" if idx == cur else "none"
        screens_html += (
            f'<div id="wd-s{idx}" style="display:{display}">'
            f'{content}</div>'
        )

    # Progress dots
    prog_labels = [
        ("📷", "we identify<br>your dish"),
        ("🫧", "we find its<br>taste fingerprint"),
        ("🍷", "we match it<br>to a wine"),
    ]
    dots_html = ""
    for i, (icon, label) in enumerate(prog_labels):
        if i < cur:
            dot_bg, dot_c, dot_border = "#f6f5f1", "#5a5855", "rgba(26,25,23,0.20)"
            lbl_c = "#5a5855"
        elif i == cur:
            dot_bg, dot_c, dot_border = "#B85C38", "#fff", "#B85C38"
            lbl_c = "#B85C38"
        else:
            dot_bg, dot_c, dot_border = "#fff", "#9a9895", "rgba(26,25,23,0.10)"
            lbl_c = "#9a9895"

        connector = (
            '<div style="flex:1;height:1px;background:rgba(26,25,23,0.10);'
            'margin-top:-20px;z-index:0"></div>'
            if i < 2 else ""
        )
        dots_html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:5px;z-index:1">
          <div onclick="wdGoTo({i})"
               style="width:28px;height:28px;border-radius:50%;display:flex;
                      align-items:center;justify-content:center;font-size:12px;
                      background:{dot_bg};border:1.5px solid {dot_border};
                      color:{dot_c};cursor:pointer;z-index:1">{icon}</div>
          <div style="font-size:10px;color:{lbl_c};text-align:center;
                      font-weight:500;line-height:1.35">{label}</div>
        </div>
        {connector}"""

    # Nav buttons
    prev_dis = "opacity:0.28;pointer-events:none" if cur == 0 else ""
    next_dis = "opacity:0.28;pointer-events:none" if cur >= unlocked else ""
    next_lbl = "done ✓" if cur == 2 else "next →"

    return f"""
<div style="font-family:'Segoe UI',Arial,sans-serif;background:#f6f5f1;
            border-radius:16px;padding:24px 28px;max-width:720px;margin:0 auto;
            box-shadow:0 1px 3px rgba(0,0,0,0.06);border:0.5px solid rgba(26,25,23,0.10)">

  <div style="display:flex;justify-content:space-between;align-items:baseline;
              padding-bottom:1.25rem;border-bottom:0.5px solid rgba(26,25,23,0.10);
              margin-bottom:1.75rem">
    <div style="font-size:19px;font-weight:600;color:#1a1917;letter-spacing:-0.3px">
      Wine<span style="color:#B85C38"> &amp; </span>Dine
    </div>
    <div style="font-size:11px;color:#9a9895;font-style:italic">
      photo &rarr; fingerprint &rarr; pairing
    </div>
  </div>

  <div style="display:flex;align-items:flex-start;margin-bottom:1.75rem;gap:0">
    {dots_html}
  </div>

  {screens_html}

  <div style="display:flex;justify-content:space-between;align-items:center;
              margin-top:1.5rem;padding-top:1rem;
              border-top:0.5px solid rgba(26,25,23,0.10)">
    <button onclick="wdGoTo({cur-1})"
            style="padding:8px 16px;border-radius:8px;border:0.5px solid rgba(26,25,23,0.20);
                   background:transparent;font-size:13px;color:#5a5855;cursor:pointer;{prev_dis}">
      &larr; back
    </button>
    <span style="font-size:11px;color:#9a9895">{cur+1} of 3</span>
    <button onclick="wdGoTo({cur+1})"
            style="padding:8px 16px;border-radius:8px;border:0.5px solid #B85C38;
                   background:#B85C38;font-size:13px;color:#fff;cursor:pointer;{next_dis}">
      {next_lbl}
    </button>
  </div>

  <div style="margin-top:14px;font-size:10px;color:#ccc;text-align:right">
    Wine &amp; Dine &middot; RSU Advanced ML &middot; 2026
  </div>
</div>

<script>
(function() {{
  var unlocked = {unlocked};
  function wdGoTo(n) {{
    if (n < 0 || n > unlocked) return;
    for (var i = 0; i < 3; i++) {{
      var s = document.getElementById('wd-s' + i);
      if (s) s.style.display = (i === n) ? 'block' : 'none';
    }}
  }}
  window.wdGoTo = wdGoTo;
}})();
</script>"""


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

    s2 = _screen2_html(display, desc, attn_w, cluster_idx, cluster_name, sims)

    # ── Yield 1: screen2 done, screen3 loading ───────────────────────────────
    yield _shell_html(s1, s2, _LOADING_SPINNER, 1)

    recs = RESULTS_ALL.get(food_key, [])
    safe_cluster = recs[0].get("name", cluster_name) if recs else cluster_name
    feel = _food_feel(safe_cluster)
    s3 = _screen3_html(display, cluster_name, recs, feel)

    # ── Yield 2: all done, advance to screen3 ───────────────────────────────
    yield _shell_html(s1, s2, s3, 2)



# ── App state ─────────────────────────────────────────────────────────────────
_state: dict = {"food": "", "conf": 0.0, "top5": []}

# ── Event handlers ─────────────────────────────────────────────────────────────
def on_identify(pil_img):
    """CNN pass — returns screen-1 shell and shows yes/no confirm row."""
    if pil_img is None:
        return "", gr.update(visible=False)

    food_name, conf, top5 = identify_food(pil_img)
    _state.update(food=food_name, conf=conf, top5=top5)

    s1   = _screen1_html(food_name, conf, top5)
    html = _shell_html(s1, "", "", 0)
    return html, gr.update(visible=True)


def on_yes():
    """BiLSTM pass — streams screens 2 then 3 into the shell."""
    food_name = _state.get("food", "")
    food_key  = food_name.lower().replace(" ", "_")
    cluster, cluster_name, sims, desc, attn_w = bilstm_encode(food_key)

    # hide confirm buttons immediately
    yield gr.update(visible=False), gr.update()

    for html in _wine_card_parts(food_name, _state["conf"], _state["top5"],
                                 cluster, cluster_name, sims, desc, attn_w):
        time.sleep(0.5)
        yield gr.update(visible=False), html


def on_no():
    """Reset to blank state."""
    _state.update(food="", conf=0.0, top5=[])
    return "", gr.update(visible=False)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="orange", secondary_hue="orange"),
    title="Wine & Dine 🍷",
    css=".gradio-container{max-width:760px !important;margin:auto}",
) as demo:

    gr.Markdown(
        "Upload a food photo — we identify it, read its flavor fingerprint,"
        " and find your perfect wine pairing."
    )

    with gr.Row():
        img_input    = gr.Image(type="pil", label="📷 Food photo", height=280)
        identify_btn = gr.Button("🔍 Identify", variant="primary", size="lg",
                                 scale=0, min_width=120)

    wine_card = gr.HTML()

    with gr.Row(visible=False) as confirm_row:
        yes_btn = gr.Button("✅  yes, that's my dish — show pairings!",
                            variant="primary")
        no_btn  = gr.Button("↩  not quite, try again",
                            variant="secondary")

    # wiring
    identify_btn.click(
        on_identify,
        inputs=[img_input],
        outputs=[wine_card, confirm_row],
    )
    yes_btn.click(
        on_yes,
        inputs=None,
        outputs=[confirm_row, wine_card],
    )
    no_btn.click(
        on_no,
        inputs=None,
        outputs=[wine_card, confirm_row],
    )

if __name__ == "__main__":
    demo.launch()
