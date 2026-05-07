import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = r'c:\Users\jolanta.stutane\Desktop\RSU_AI\DL_Final\wine-dine\wine-dine.ipynb'
with open(nb_path, encoding='utf-8-sig') as f:
    nb = json.load(f)

NEW_SRC = r'''# ── 14.5  recommend_v2() + print_card() ──────────────────────────────────────
import string as _str_mod
from rich.console import Console
from rich.panel   import Panel
from rich.text    import Text

_console = Console(width=55)

_punct_14 = str.maketrans("", "", _str_mod.punctuation)

def _tokenise_14(text: str):
    """Match §7 tokeniser: lowercase, strip punctuation, whitespace split."""
    return str(text).lower().translate(_punct_14).split()


def _encode_text(text: str) -> np.ndarray:
    """Encode one text string → L2-normalised (512,) numpy array via taste_encoder."""
    toks = _tokenise_14(text)
    ids  = [VOCAB.get(w, 1) for w in toks[:MAX_SEQ_LEN]]
    if not ids:
        ids = [1]
    ids += [0] * (MAX_SEQ_LEN - len(ids))
    seq  = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    lens = torch.tensor([(seq != 0).sum().item()]).clamp(min=1)
    taste_encoder.eval()
    with torch.no_grad():
        vec = taste_encoder.encode(seq, lens).cpu().numpy()[0]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _nearest_cluster(vec: np.ndarray) -> int:
    """Return index of the cluster centroid with highest cosine similarity."""
    return int(np.argmax(centroids @ vec))


def recommend_v2(food_name: str) -> list:
    """
    Return three wine recommendations for the given food.

    food_flavor_table_v2 keys (from food_flavor_description_v2.json):
      "classic"  -> SAFE BET  : the dish's dominant flavor -- closest taste match
      "safe_bet" -> HIDDEN GEM: compatible but surprising flavor
      "contrast" -> BOLD MOVE : not encoded; chosen geometrically as the cluster
                                centroid farthest from the SAFE BET centroid

    Tie-breaking: if two tiers resolve to the same cluster, the lower-priority
    tier shifts to its next-best candidate.
    """
    entry = food_flavor_table_v2.get(food_name, {})
    if not entry:
        raise KeyError(f"'{food_name}' not in food_flavor_table_v2")

    safe_desc   = entry.get("classic",  "balanced complex")
    hidden_desc = entry.get("safe_bet", "surprising compatible")

    safe_vec  = _encode_text(safe_desc)
    hid_vec   = _encode_text(hidden_desc)

    safe_k   = _nearest_cluster(safe_vec)
    hidden_k = _nearest_cluster(hid_vec)

    # Bold Move: cluster centroid farthest (Euclidean) from SAFE BET centroid
    dists  = np.linalg.norm(centroids - centroids[safe_k], axis=1)
    bold_k = int(np.argmax(dists))

    # Resolve collisions
    used = {safe_k}
    if hidden_k in used:
        sims_h = centroids @ hid_vec
        sims_h[list(used)] = -np.inf
        hidden_k = int(np.argmax(sims_h))
    used.add(hidden_k)
    if bold_k in used:
        dists2 = dists.copy()
        dists2[list(used)] = -np.inf
        bold_k = int(np.argmax(dists2))

    results = []
    for tier, icon, k in [
        ("SAFE BET",   "\u2705", safe_k),
        ("HIDDEN GEM", "\U0001f48e", hidden_k),
        ("BOLD MOVE",  "\U0001f525", bold_k),
    ]:
        w = cluster_wines[k]
        results.append({
            "tier"    : tier,
            "icon"    : icon,
            "cluster" : k,
            "name"    : cluster_names[k],
            "keywords": cluster_keywords[k],
            "wine"    : w["wine_label"],
            "rating"  : w["rating_pct"],
            "snippet" : w["review_text"],
        })
    return results


# ── Card text helpers ─────────────────────────────────────────────────────────

# Maps each user-friendly axis adjective → how that character FEELS in a food.
# Used in "As your [food] is [feel]" — derived from the SAFE BET cluster so
# it always describes the food's dominant character in taste terms, not
# cooking methods or ingredient names.
_AXIS_FOOD_FEEL = {
    "rich":     "rich and full-bodied",
    "crispy":   "bright and zesty",
    "bold":     "hearty and bold",
    "juicy":    "fruity and fresh",
    "deep":     "deep and ripe",
    "earthy":   "savory and earthy",
    "sweet":    "sweet and indulgent",
    "smoky":    "warm and smoky",
    "delicate": "light and fragrant",
    "stony":    "clean and mineral",
}

def _cluster_adj(cluster_name: str) -> str:
    """'juicy & smoky' -> 'juicy'   |   'earthy' -> 'earthy'"""
    first = cluster_name.split("&")[0].strip()
    if first.lower().startswith("the "):
        first = first[4:]
    return first.lower()

def _food_feel(safe_bet_cluster_name: str) -> str:
    """
    Derive how the food tastes from the SAFE BET cluster's dominant axis.
    Always returns a taste phrase, never a cooking method or ingredient.
    e.g. safe bet cluster "earthy & smoky" -> "savory and earthy"
    """
    adj = _cluster_adj(safe_bet_cluster_name)
    return _AXIS_FOOD_FEEL.get(adj, "rich and complex")

def _clip(text: str, max_chars: int = 150) -> str:
    """Truncate at a word boundary; never cut mid-word."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;: ") + "..."

# Intent phrase -- the one phrase that distinguishes the three tiers
_INTENT = {
    "SAFE BET":   "matches it",
    "HIDDEN GEM": "surprises you",
    "BOLD MOVE":  "goes against it",
}

# Border colour per tier
_TIER_STYLE = {
    "SAFE BET":   "green",
    "HIDDEN GEM": "blue",
    "BOLD MOVE":  "red",
}


def _build_tier_text(r: dict, display_name: str, feel: str) -> Text:
    """Compose the rich Text block for one tier panel."""
    intent  = _INTENT[r["tier"]]
    adj     = _cluster_adj(r["name"])
    kws     = "  \u00b7  ".join(r["keywords"])
    snippet = _clip(r["snippet"])

    t = Text()
    t.append(f"{r['icon']}  {r['tier']}\n\n", style="bold")
    t.append(f"As your {display_name} is {feel},\n")
    t.append("we believe you need a wine that\n")
    t.append(f"{intent}", style="bold")
    t.append(" \u2014 something ")
    t.append(f"{adj}\n", style="bold italic")
    t.append("that plays on\n")
    t.append(f"{kws}\n", style="bold cyan")
    t.append("\n")
    t.append("Wine drinkers suggest:\n", style="dim")
    t.append(f"{r['wine']}  \u2b50 {r['rating']}\n", style="bold green")
    t.append("\n")
    t.append("Wine drinkers say:\n", style="dim")
    t.append(f'"{snippet}"', style="italic")
    return t


def print_card(food_name: str, recs: list):
    """Print a story-driven recommendation card for one food using rich."""
    display_name = food_name.replace("_", " ").title()

    # Derive food feel from the SAFE BET cluster (always recs[0])
    feel = _food_feel(recs[0]["name"])

    # Food title panel
    _console.print(Panel(
        f"[bold]\U0001f37d   {display_name}[/bold]",
        border_style="white",
    ))

    # One panel per tier
    for r in recs:
        body = _build_tier_text(r, display_name, feel)
        _console.print(Panel(
            body,
            border_style=_TIER_STYLE[r["tier"]],
            padding=(0, 1),
        ))

    _console.print()


# ── Quick sanity test (3 foods) ───────────────────────────────────────────────
for _food in ["pizza", "sushi", "steak"]:
    try:
        _recs = recommend_v2(_food)
        print_card(_food, _recs)
    except KeyError as exc:
        _console.print(f"  \u26a0  {exc}")

print("\u2713 Section 14.5 complete \u2014 recommend_v2() and print_card() ready.")
'''

def to_source(code):
    lines = code.split('\n')
    result = [l + '\n' for l in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result

changed = 0
for cell in nb['cells']:
    if cell.get('id') == '5a710aff':
        cell['source'] = to_source(NEW_SRC)
        cell['outputs'] = []
        cell['execution_count'] = None
        changed += 1
        print('  ✓ Patched §14.5')
        break

if changed:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Saved.')
else:
    print('WARNING: cell not found.')
