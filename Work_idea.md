# Wine Peer

## Advanced Machine Learning · Final Project

---

## Product description

Wine Peer is a wine recommendation app built for anyone sitting down to a meal. The question "what wine goes with this?" is common but rarely answered well — most people default to habit or ask a sommelier they don't have. Wine Peer makes expert pairing knowledge available through a single photograph.

The interaction takes under 10 seconds: photograph your food, get back three recommended grape varieties — one that complements the food's dominant flavor, one that contrasts it, and one that balances it — each with a tasting description in the words of a real Vivino reviewer, a specific wine bottle with vintage, and a user approval percentage.

---

## How it works

Four layers in sequence — each independent, each testable on its own:

```text
PHOTO  ->  CNN  ->  food label  ->  food flavor profile  ->  Word2Vec similarity  ->  grape varieties (Complement / Contrast / Balance)  ->  BiLSTM review retrieval  ->  Vivino tasting note + wine + rating %
```

| Layer | What it does | ML involved |
| --- | --- | --- |
| CNN | Classifies the food photo into one of 101 categories | Yes — image classification |
| Food flavor table | Maps the food label to three sets of flavor keywords (complement / contrast / balance) | No — curated dict, embedded in notebook |
| Word2Vec pairing | Pre-trained on Google News (knows food words: *tomato*, *fatty*, *smoky*), then fine-tuned on 824k Vivino reviews (adds wine words: *Sangiovese*, *tannic*, *cassis*). Cosine similarity between food flavor keywords and grape embeddings finds the best grape per intent. | Yes — transfer learning + fine-tuning |
| BiLSTM | Trained on WineSensed reviews for 15-class grape variety classification; at inference, retrieves the most representative real Vivino review per recommended grape + highest-rated wine of that grape | Yes — text classification + encoder retrieval |

The food flavor table is the bridge. It is a Python dict with 101 entries, each containing three keyword sets. Because Word2Vec starts from Google News, the flavor table can use natural food language — *tomato*, *fatty*, *smoky*, *creamy* — without needing to restrict itself to wine vocabulary.

---

## Example output

| | |
| --- | --- |
| **Input** | Photo of grilled salmon |
| **CNN output** | Grilled Salmon (91% confidence) |
| **Food flavor profile** | umami · fatty · smoky · savory · delicate · ocean |

| Pairing | Grape Variety | Wine | Vivino % | Tasting Note |
| --- | --- | --- | --- | --- |
| **Complement** — echoes the food's richness | Chardonnay | *Domaine Leflaive Puligny-Montrachet 2021* | 94% | *"Creamy texture with toasted oak and ripe stone fruit — beautiful alongside fatty fish."* |
| **Contrast** — cuts through and refreshes | Sauvignon Blanc | *Cloudy Bay Te Koko 2022* | 91% | *"Bright citrus and grassy minerality. The acidity lifts oily salmon beautifully."* |
| **Balance** — elegant crowd pleaser | Pinot Noir | *A to Z Wineworks Oregon Pinot Noir 2021* | 89% | *"Silky red cherry and forest floor. Light enough not to overpower delicate fish."* |

---

## Outputs

| Output | How it is produced |
| --- | --- |
| **Food category** | CNN classifies the photo (101 Food-101 classes) |
| **Complement pairing** | Word2Vec: food complement-keywords -> cosine similarity -> closest grape variety |
| **Contrast pairing** | Word2Vec: food contrast-keywords -> cosine similarity -> closest grape variety |
| **Balance pairing** | Word2Vec: food balance-keywords -> cosine similarity -> closest grape variety |
| **Wine name + vintage** | BiLSTM retrieval: highest-rated `wine_label` from WineSensed for the predicted grape |
| **Vivino approval %** | `rating_pct = rating / 5 * 100` from WineSensed row |
| **Tasting note** | BiLSTM encoder: most representative real Vivino review per grape |
| **Compatibility score** *(+10 bonus)* | Joint model: CNN image embedding + BiLSTM review embedding -> compatible / not |

---

## Datasets

### Image dataset — Food-101

`torchvision.datasets.Food101(root=DATA_DIR, split="train", download=True)`

- **101,000 images** across 101 food categories (750 train + 250 test per class)
- Clean, pre-labeled, loads in one line — no preprocessing required
- License: ETH Zurich research dataset
- **CNN task:** 101-class food classification

### Text dataset — WineSensed (Vivino reviews)

`load_dataset("Dakhoo/L2T-NeurIPS-2023", "vintages", trust_remote_code=True)`

- **824,000 real Vivino tasting notes** written by wine community members
- Columns: `review_text`, `grape`, `wine`, `year`, `rating`, `country`, `region`
- **BiLSTM task:** classify review text into 15 grape variety classes
- **Word2Vec task:** fine-tune pre-trained Google News Word2Vec on review text — bridges general food vocabulary to wine tasting language in a single shared vector space
- License: CC BY-NC-ND 4.0

### Pairing table — Food flavor table

- Python dict embedded directly in the notebook (Section 2.4) — no file dependency
- 101 entries, one per Food-101 class
- Each entry has three keyword sets drawn from wine tasting vocabulary:

| Key | Purpose | Example for `grilled_salmon` |
| --- | --- | --- |
| `complement` | Flavors that echo and amplify the food | `creamy oaky buttery rich stone-fruit toasty` |
| `contrast` | Flavors that cut through and refresh | `citrus mineral grassy crisp acidic zesty` |
| `balance` | Elegant, crowd-safe, neither overwhelms | `silky red-fruit light earthy cherry delicate` |

---

## Grape variety classes (top 15)

WineSensed contains hundreds of grape varieties. Top 15 by review frequency cover ~85% of all reviews:

| Class | Grape Variety | Color |
| --- | --- | --- |
| 0 | Cabernet Sauvignon | Red |
| 1 | Merlot | Red |
| 2 | Pinot Noir | Red |
| 3 | Syrah | Red |
| 4 | Malbec | Red |
| 5 | Sangiovese | Red |
| 6 | Tempranillo | Red |
| 7 | Grenache | Red |
| 8 | Zinfandel | Red |
| 9 | Chardonnay | White |
| 10 | Sauvignon Blanc | White |
| 11 | Riesling | White |
| 12 | Pinot Grigio | White |
| 13 | Viognier | White |
| 14 | Chenin Blanc | White |

---

## Models

### CNN — food classification from photo

- **Input:** food photograph (224x224 RGB)
- **Task:** 101-class food classification
- **Architecture 1:** Custom CNN trained from scratch (>=3 conv blocks)
- **Architecture 2:** ResNet-50, frozen backbone + fine-tuned head
- **Explainability:** Grad-CAM — which part of the food photo drove the prediction?

### BiLSTM — grape variety classification from Vivino review text

- **Input:** WineSensed review text
- **Task:** 15-class grape variety classification (random baseline = 6.7%)
- **Architecture 1:** Unidirectional LSTM baseline
- **Architecture 2:** Bidirectional LSTM with attention
- **Embeddings:** GloVe-100d pre-trained word vectors
- **Explainability:** attention weight visualisation — which words (*"cassis"*, *"mineral"*, *"floral"*) drove the grape prediction?
- **Inference mode:** grape-conditioned retrieval — surface the most representative Vivino review + highest-rated wine per grape
- **Standalone text mode:** user pastes a tasting note -> BiLSTM predicts the grape variety with confidence

### Joint Model — food-wine compatibility *(+10 bonus points)*

- **Positive pairs:** (food image, wine review of grape suggested by flavor table) — label `1`
- **Negative pairs:** random (food image, wine review of clearly incompatible grape) — label `0`
- **Architecture:** frozen CNN encoder (2048-d) + frozen BiLSTM encoder (256-d) -> concat (2304-d) -> FC -> compatible/not
- Train only the FC head; encoders are frozen
- **Value:** generalises beyond the flavor table — can score food-grape pairs not hand-coded; interesting experiment: print top-5 unexpected high-scoring pairs

---

## Business integration

The recommendation card returned to the user:

1. **Food identified** — "Grilled Salmon" (CNN, 91% confidence) + Grad-CAM heatmap
2. **Complement pairing** — Chardonnay + wine bottle + Vivino % + real tasting note
3. **Contrast pairing** — Sauvignon Blanc + wine bottle + Vivino % + real tasting note
4. **Balance pairing** — Pinot Noir + wine bottle + Vivino % + real tasting note
5. **Joint model score** *(bonus)* — compatibility confidence for each pairing

**Target users:** restaurant guests, home cooks, wine shop staff, event planners. Primary demo: restaurant staff recommending wines to guests without a sommelier on duty.

**What makes this better than a hardcoded list:** the pairings emerge from 824k real Vivino tasting notes via Word2Vec; the quotes are genuine community language retrieved by the BiLSTM; the wine bottles and ratings come from real Vivino data — nothing is written by hand.

---

## Submission deliverables

| # | What | Details |
| --- | --- | --- |
| 1 | `beer_peer.ipynb` | Single notebook, all code and outputs, runs end to end |
| 2 | `beer_peer.pdf` | Same notebook exported as PDF |
| 3 | Presentation | 15-20 min slide deck (.pptx or .pdf) |
| 4 | Deployment link | Wine Peer hosted on **Hugging Face Spaces** (Streamlit) |

---

## File structure

```text
beer-peer/
├── beer_peer.ipynb             <- the submission — fully self-contained
├── requirements.txt
├── weights/
│   ├── cnn_scratch.pt
│   ├── cnn_resnet50.pt
│   ├── lstm.pt
│   ├── bilstm.pt
│   └── joint_model.pt
├── figures/                    <- all saved plots (created by notebook)
└── deployment/
    └── app.py                  <- Streamlit app (single file)
```

---

## Notebook structure

| Section | Content |
| --- | --- |
| 1 | Environment setup — dependencies and imports |
| 2 | Data loading — Food-101 + WineSensed + pairing table |
| 3 | EDA — image dataset (class distribution, sample grid) |
| 4 | EDA — text dataset (review length, grape distribution, word clouds per variety) |
| 5 | Image preprocessing and data loaders |
| 6 | Text preprocessing and data loaders |
| 7 | CNN — custom architecture (trained from scratch) |
| 8 | CNN — ResNet-50 (transfer learning) |
| 9 | CNN explainability — Grad-CAM |
| 10 | LSTM — unidirectional baseline |
| 11 | BiLSTM — bidirectional with attention |
| 12 | BiLSTM explainability — attention weights |
| 13 | Joint model — food-wine compatibility classifier *(+10 bonus)* |
| 14 | Business integration — recommendation card, 20-example table |
| 15 | Business framing, ethics, and team contributions |

---

## Development workflow

**Phase 1 — VS Code (local, CPU)**
Sections 1-4, 6, 10-12. Data loading, EDA, text preprocessing, LSTM + BiLSTM training. All CPU-friendly.

**Phase 2 — Google Colab (GPU)**
Sections 5, 7-9, 13. Image data loaders, CNN training, joint model. Save weights to Drive.

**Phase 3 — VS Code (local, CPU)**
Load saved weights. Grad-CAM, attention, recommendation card, Streamlit app, PDF export.

---

## One-sentence pitch

> "Wine Peer turns a food photo into a wine recommendation — combining computer vision, natural language understanding, and 824k real Vivino tasting notes."

---

Dataset licenses: Food-101 (ETH Zurich, research use) · WineSensed (CC BY-NC-ND 4.0, non-commercial)
