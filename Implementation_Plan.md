# Wine Peer — Implementation Plan

## Step-by-step reference

> **Architecture in one sentence:** CNN classifies a food photo → food label → food flavor table (complement / contrast / balance keyword sets in wine vocabulary) → Word2Vec cosine similarity → 3 grape variety recommendations → BiLSTM encoder retrieves the most representative real Vivino review per grape + highest-rated wine of that grape; joint model learns food-wine compatibility from (CNN image embedding, BiLSTM review embedding) pairs.

---

## Product recap

A user photographs their food. Wine Peer returns three wine recommendations — one that complements the food's flavor, one that contrasts it, and one that balances it — each with a real Vivino tasting note, a wine bottle name and vintage, and a user approval percentage. Three independent components: a CNN that sees the food, Word2Vec that bridges food flavor vocabulary to grape variety vocabulary, and a BiLSTM that retrieves genuine Vivino review sentences per grape. The joint model (+10 pts) learns food-wine compatibility at the feature level — scoring (image embedding, review embedding) pairs.

---

## Datasets

| Dataset | How to load | Content | Used for |
| --- | --- | --- | --- |
| Food-101 | `torchvision.datasets.Food101` | 101,000 images, 101 food classes | CNN training |
| WineSensed | `load_dataset("Dakhoo/L2T-NeurIPS-2023", "vintages", trust_remote_code=True)` | 824k real Vivino reviews, `grape`, `wine`, `year`, `rating`, `review` | BiLSTM training · Word2Vec training · review retrieval |
| Food flavor table | Embedded Python dict in notebook (Section 2.4) | 101 foods × 3 keyword sets (`complement` / `contrast` / `balance`) | Word2Vec pairing · joint model labels |

**BiLSTM labels:** top 15 grape varieties by review count (Cabernet Sauvignon / Merlot / Pinot Noir / Syrah / Malbec / Sangiovese / Tempranillo / Grenache / Zinfandel / Chardonnay / Sauvignon Blanc / Riesling / Pinot Grigio / Viognier / Chenin Blanc) — covering ~85% of all reviews.

---

## Development phases

| Phase | Environment | Sections | What happens |
| --- | --- | --- | --- |
| **Phase 1** | VS Code (CPU) | 1, 2, 3, 4, 6, 10, 11 | Environment setup. Data loading. EDA (images + text). Text preprocessing + data loaders. LSTM baseline. BiLSTM with attention. |
| **Phase 2** | Google Colab (GPU) | 5, 7, 8, 13 | Image preprocessing + data loaders. CNN scratch. CNN ResNet-50. Joint model (full training). |
| **Phase 3** | VS Code (CPU) | 9, 12, 14, 15, deployment | Grad-CAM explainability. BiLSTM attention explainability. Recommendation card + 20-example table. Business framing + ethics. HF Spaces deployment. |

---

## Before you start — one-time setup

1. Folders `weights/`, `figures/`, `deployment/` are created automatically by the notebook.
2. Open `beer-peer/beer_peer.ipynb`.
3. Run **Section 1** (pip installs) once per environment. No Kaggle API key needed — WineSensed loads via Hugging Face.

---

## PHASE 1 — Local skeleton (VS Code, CPU)

> Goal: every cell runs without errors. Accuracy does not matter yet.

---

### STEP 1 — Load the datasets

**Section 2 of the notebook.**

```python
import torchvision.datasets as tv_datasets
ds_train = tv_datasets.Food101(root=DATA_DIR, split="train", download=True)
ds_test  = tv_datasets.Food101(root=DATA_DIR, split="test",  download=True)

from datasets import load_dataset
_ds = load_dataset("Dakhoo/L2T-NeurIPS-2023", "vintages", trust_remote_code=True)
df_wine = pd.concat([_ds[s].to_pandas() for s in _ds.keys()], ignore_index=True)
```

- Print shapes, column names, class counts for both.
- Extract primary grape (first entry of `grape` column); select top 15 by frequency; add `grape_class` column; print distribution.
- Compute `rating_pct = rating / 5.0 × 100`; build `wine_label = wine + year`.
- Load the food flavor table (Section 2.4 embedded dict); display first 5 entries as a DataFrame with columns `food`, `complement`, `contrast`, `balance`.

**Done when:** Food-101 dataset objects, WineSensed DataFrame, and food flavor table all visible with correct sizes.

---

### STEP 2 — EDA: image dataset

**Section 3 of the notebook.**

- 3–6 sample image grid (2 images per food class, randomly sampled from 18 classes).
- Class distribution bar chart (101 bars — confirm ~750 train / ~250 test per class).
- Image dimension histogram (confirm variable resolution ? need resize in preprocessing).

**Done when:** sample grid and distribution chart visible.

---

### STEP 3 — EDA: text dataset

**Section 4 of the notebook.**

- Review length histogram (words per review); mark the 95th percentile → set as max token length.
- Grape class distribution bar chart (15 bars — confirm no extreme imbalance).
- Word cloud per grape variety (15 clouds — visually confirm that each grape has distinct vocabulary).
- 3 sample reviews per grape variety.

**Done when:** all plots visible; max token length chosen.

---

### STEP 4 — Text preprocessing, Word2Vec, and data loaders

**Section 6 of the notebook.**

- Tokenise `review_text` at word level; lowercase; strip punctuation.
- Load GloVe-100d for BiLSTM; build vocab from training split only (no data leakage).
- Pad/truncate to 95th-percentile token length; print % truncated.
- Load pre-trained Google News Word2Vec via gensim (`api.load("word2vec-google-news-300")`); this gives the model general food vocabulary (*tomato*, *fatty*, *smoky*) before any wine-specific training.
- Fine-tune on all WineSensed review text: `w2v.build_vocab(wine_sentences, update=True)` → `w2v.train(wine_sentences, total_examples=len(wine_sentences), epochs=5)`. This anchors wine-specific words (*Sangiovese*, *cassis*, *tannic*, *terroir*) into the shared space (~10–15 minutes CPU).
- Save fine-tuned model to `weights/w2v_finetuned.model`.
- Compute grape embeddings: for each of 15 grape varieties, average all word vectors across reviews for that grape → 15 grape vectors. Save to `weights/grape_embeddings.npy`.
- **Note:** add `~/gensim-data/` to `.gitignore` — the base Google News model (~1.6 GB) must not be committed.
- Train / val / test split: 70 / 15 / 15, stratified by `grape_class`, `SEED=42`.
- Create BiLSTM `DataLoader` (batch size 64).

**Done when:** one BiLSTM batch prints `torch.Size([64, N])`; 15 grape vectors saved.

---

### STEP 5 — CNN skeleton (architecture only)

**Section 7 of the notebook.**

- Define 3-block custom CNN: `Conv2d → BatchNorm → ReLU → MaxPool2d` × 3 → `Flatten → FC(512) → FC(101)`.
- `model(torch.randn(2, 3, 224, 224))` returns `[2, 101]`.
- Full training in Phase 2 only.

**Done when:** forward pass runs; parameter count printed.

---

### STEP 6 — ResNet-50 skeleton

**Section 8 of the notebook.**

- `torchvision.models.resnet50(weights='IMAGENET1K_V1')`; freeze all; replace `fc` with `nn.Linear(2048, 101)`.
- Forward pass returns `[2, 101]`.

**Done when:** shape correct; frozen vs. trainable parameter counts printed.

---

### STEP 7 — LSTM baseline (full training on CPU)

**Section 10 of the notebook.**

- `nn.Embedding(GloVe) → nn.LSTM(hidden=128) → nn.Linear(128, 15)`.
- Train 3 epochs; plot loss + accuracy curves.
- Report val accuracy (random baseline = 6.7% across 15 classes).

**Done when:** loss decreases across epochs.

---

### STEP 8 — BiLSTM with attention (full training on CPU)

**Section 11 of the notebook.**

- `bidirectional=True`, hidden 128 (output 256).
- Attention: `score = softmax(W · h_t)` → weighted sum → classification head.
- Train 3 epochs; compare val accuracy vs. LSTM baseline.
- Save `weights/bilstm.pt`.

**Done when:** BiLSTM trained, weights saved, accuracy comparison written in markdown.

---

### STEP 9 — Joint model skeleton

**Section 13 of the notebook.**

Build the training data from what you already have:

```python
# Positive pairs: food image + review of a grape whose Word2Vec similarity score
# to any of the food's keyword sets (complement/contrast/balance) exceeds threshold → label 1
# Negative pairs: food image + review of a clearly incompatible grape → label 0

def build_pairs(food_flavor_table, df_wine, grape_embeddings, w2v_model,
                n_negatives_per_positive=3):
    pairs = []
    for food, profiles in food_flavor_table.items():
        food_images = get_food_images(food)  # from Food-101
        compatible_grapes = set()
        for intent in ["complement", "contrast", "balance"]:
            vec = embed_keywords(profiles[intent], w2v_model)
            top_grape = cosine_top1(vec, grape_embeddings)
            compatible_grapes.add(top_grape)

        for grape in compatible_grapes:
            reviews = df_wine[df_wine["grape_class"] == grape]["review_text"].sample(5)
            for img in food_images[:5]:
                for rev in reviews:
                    pairs.append((img, rev, 1))  # positive

        incompatible = set(GRAPE_CLASSES) - compatible_grapes
        for grape in random.sample(incompatible, n_negatives_per_positive):
            reviews = df_wine[df_wine["grape_class"] == grape]["review_text"].sample(5)
            for img in food_images[:5]:
                for rev in reviews:
                    pairs.append((img, rev, 0))  # negative
    return pairs
```

Architecture (skeleton — forward pass on dummy tensors only):

```python
class CompatibilityModel(nn.Module):
    def __init__(self, cnn_encoder, bilstm_encoder):
        super().__init__()
        self.cnn = cnn_encoder    # frozen, outputs 512-d (scratch) or 2048-d (ResNet)
        self.bilstm = bilstm_encoder  # frozen, outputs 256-d
        self.fc = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, text):
        img_emb = self.cnn(image)    # [B, 512]
        txt_emb = self.bilstm(text)  # [B, 256]
        x = torch.cat([img_emb, txt_emb], dim=1)  # [B, 768]
        return self.fc(x)            # [B, 1]
```

- `model(torch.randn(2,3,224,224), torch.randint(0,1000,(2,N)))` returns `[2, 1]`.
- Full training in Phase 2.

**Done when:** forward pass runs; only FC head parameters are trainable.

---

## PHASE 2 — Real training (Google Colab, GPU)

> Goal: get good weights. Only this phase requires GPU.

---

### STEP 10 — Move to Colab

- Upload `beer-peer/beer_peer.ipynb` to Colab; mount Drive; set weight output paths to Drive.
- Run **Section 5**: build image `DataLoader` from Food-101.
  - Transforms: `Resize(256) ? CenterCrop(224) ? RandomHorizontalFlip ? ColorJitter ? ToTensor ? Normalize(ImageNet stats)`.
  - Train / val / test split stratified by class (750/250 already done by Food-101 — use official split).
  - Confirm one batch: `torch.Size([B, 3, 224, 224])`.

---

### STEP 11 — Train CNN scratch (full)

**Section 7 — full training.**

- 20 epochs, early stopping on val loss (patience 3).
- Optimizer: Adam lr 1e-3; scheduler: ReduceLROnPlateau.
- Save best to `weights/cnn_scratch.pt`.
- Report test: Accuracy (top-1 and top-5), macro F1, Confusion Matrix (sample 20 classes for readability).

---

### STEP 12 — Train ResNet-50 (full)

**Section 8 — full training.**

- Phase A: frozen backbone, train FC head only, 5 epochs, lr 1e-3.
- Phase B: unfreeze last ResNet block (`layer4`), lr 1e-4, 10 more epochs.
- Save best to `weights/cnn_resnet50.pt`.
- Report test: Accuracy, macro F1, Confusion Matrix.
- Markdown: compare scratch vs. ResNet-50 — accuracy, training time, top-5 failure modes.

---

### STEP 13 — Train joint model (full)

**Section 13 — full training.**

- Load frozen `cnn_resnet50.pt` encoder (replace FC with identity to get 2048-d embedding).
- Load frozen `bilstm.pt` encoder (remove classification head to get 256-d embedding).
- Build compatibility pairs (Step 9 code, now on full data).
- Train only the FC head, 10 epochs, BCELoss, Adam lr 1e-3.
- Save to `weights/joint_model.pt`.
- Report: AUC-ROC, precision, recall on test pairs.
- **The interesting experiment:** run the joint model on food-grape pairs NOT in the flavor table. Print the top-5 unexpected high-scoring pairs — these are the "learned extensions" beyond the hand-curated rules.

---

### STEP 14 — Download artefacts

Download from Colab / Drive to local `weights/`:

- `cnn_scratch.pt`
- `cnn_resnet50.pt`
- `bilstm.pt`
- `joint_model.pt`

---

## PHASE 3 — Analysis, explainability, deployment (VS Code, CPU)

> Goal: rubric-required analysis and working demo.

---

### STEP 15 — CNN explainability (Grad-CAM)

**Section 9 of the notebook.**

- Load `cnn_resnet50.pt`; select 6 test images (2 correct, 2 wrong class, 2 top-5 correct but top-1 wrong).
- Generate Grad-CAM heatmap with `torchcam`; display original + overlay side by side.
- 3-sentence markdown per example: which part of the food drove the classification?

---

### STEP 16 — BiLSTM explainability (attention weights)

**Section 12 of the notebook.**

- Load `bilstm.pt`; pick 6 reviews (1 per grape variety, sampled from the 15 classes).
- Extract attention weight vector; render as colour-highlighted text (word opacity ∝ weight).
- 3-sentence interpretation: which words (*"cassis"*, *"mineral"*, *"floral"*) drove each grape prediction?

---

### STEP 17 — Recommendation card + 20-example table

**Section 14 of the notebook.**

```python
def recommend(image_path):
    # 1. CNN → food label (top-3 with confidence)
    food_label, confidence = cnn_predict(image_path)

    # 2. Food flavor table → three keyword sets
    profiles = food_flavor_table[food_label]  # {complement: [...], contrast: [...], balance: [...]}

    # 3. Word2Vec → one grape variety per pairing intent
    recommendations = {}
    for intent in ["complement", "contrast", "balance"]:
        vec = embed_keywords(profiles[intent], w2v_model)
        recommendations[intent] = cosine_top1(vec, grape_embeddings)

    # 4. Joint model → re-score each recommended grape (optional reranking)
    joint_scores = {
        intent: joint_score(image_path, sample_review(grape))
        for intent, grape in recommendations.items()
    }

    # 5. BiLSTM → retrieve most representative Vivino review per grape
    #    + fetch highest-rated real wine of that grape
    flavor_text = {}
    wine_names   = {}
    rating_pcts  = {}
    for intent, grape in recommendations.items():
        review, wine_label, pct = get_representative_review(
            grape, bilstm_encoder, df_wine
        )
        flavor_text[intent] = review
        wine_names[intent]  = wine_label
        rating_pcts[intent] = pct

    return format_card(
        food_label, confidence, recommendations,
        wine_names, rating_pcts, joint_scores, flavor_text
    )
```

- Run on 20 Food-101 test images.
- Display as table: Food | CNN confidence | Grape 1 (Complement) | Grape 2 (Contrast) | Grape 3 (Balance) | Wine names | Rating % | Joint scores.
- Highlight 3 cases where joint model re-ranked the flavor table order — explain why.

---

### STEP 18 — Business framing (markdown cells)

**Section 15 of the notebook.**

Three markdown cells:

1. **Business framing** — target user (restaurant guests, home cooks, event planners, wine shops). Decision supported: which wine to order, recommend, or stock. Cost of a bad recommendation: wasted purchase or disappointed guest. Cost of a missed pairing: guest defaults to habit instead of discovering something new.
2. **Ethics and bias** — WineSensed / Vivino skews toward European wines and English-language reviewers. Food-101 skews toward Western dishes. The food flavor table is curated from one cultural perspective. The joint model inherits all these biases. Mitigation ideas: multilingual reviews, regional dataset expansion, diverse flavor table curators.
3. **Team contribution table** — each member's section ownership.

---

### STEP 19 — Hugging Face Spaces deployment

**`deployment/app.py`**

Two-column layout:

- **Left:** Upload food photo → CNN food label + confidence bar + Grad-CAM heatmap.
- **Right:** Three wine recommendation cards, each showing: grape variety name, wine bottle + vintage, Vivino approval %, real tasting note from BiLSTM-retrieved Vivino review, joint compatibility score.
- **Standalone text mode:** User can also paste a Vivino-style tasting note → BiLSTM predicts the grape variety with confidence (satisfies rubric requirement for standalone text input).

**One-time setup (before Phase 3):**

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Create a new Space: `Jolanati/wine-peer`, SDK = **Streamlit**.
3. Clone the Space repo locally:

   ```bash
   git clone https://huggingface.co/spaces/Jolanati/wine-peer
   ```

4. Enable Git LFS for model weights:

   ```bash
   git lfs install
   git lfs track "*.pt"
   git add .gitattributes
   ```

**Deploy (after weights are trained in Phase 2):**

```bash
# copy app.py, requirements.txt, weights/ into the cloned Space repo
git add .
git commit -m "deploy wine-peer"
git push
```

HF builds and hosts automatically — public URL is the submission link.

Test locally first: `streamlit run deployment/app.py`

**`requirements.txt` for the Space:**

```text
streamlit
torch
torchvision
nltk
Pillow
pandas
```

---

## Final checklist before submission

- [ ] Notebook runs end to end without errors (Restart Kernel → Run All)
- [ ] All 15 sections have visible outputs
- [ ] `figures/` contains all saved plots
- [ ] `weights/` contains all four `.pt` files
- [ ] `recommend()` runs on a new food photo end to end and shows grape + wine + tasting note + rating %
- [ ] Standalone text mode: paste a tasting note → BiLSTM predicts grape variety
- [ ] Joint model "unexpected pairings" experiment has visible output
- [ ] Streamlit app runs locally (`streamlit run deployment/app.py`)
- [ ] Weights pushed to HF Space via Git LFS (`git lfs track "*.pt"`)
- [ ] HF Space builds without errors and public URL is accessible
- [ ] Deployment link works (submit HF Space URL)
- [ ] Notebook exported as PDF
- [ ] Presentation covers: business problem, EDA, CNN results, BiLSTM results, joint model + unexpected pairings, Grad-CAM, attention, deployment demo, ethics

---

Total steps: 19 | Phases: 3 | GPU required: Phase 2 only

| Phase | Where | Sections | Needs GPU |
| --- | --- | --- | --- |
| 1 | VS Code, CPU | 1–4, 6, 10–11, 13 (skeleton) | No |
| 2 | Google Colab | 5, 7–8, 13 (full) | **Yes** |
| 3 | VS Code, CPU | 9, 12, 14–15, deployment | No |
